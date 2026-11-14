"""Treinamento da PINN com busca de hiperparâmetros (Optuna) e GroupKFold.

* O agrupamento é por ``alloy_id`` (mesma composição, rotas distintas): uma
  liga nunca aparece simultaneamente em treino e validação, o que mede a
  capacidade de *extrapolação composicional* — exatamente o que o design
  inverso exige.
* As estatísticas de normalização são ajustadas apenas no fold de treino.
* O modelo final é treinado em todos os dados pelo número médio de épocas
  ótimas observado na validação cruzada.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from src.config import DEFAULT_MODEL_PATH, TrainingConfig
from src.data_pipeline.preprocessing import NormalizationStats, ProcessedDataset
from src.models.pinn import (
    PhysicsInformedNet,
    PhysicsWeights,
    PINNConfig,
    predict,
    sample_collocation_points,
    save_checkpoint,
    total_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class HyperParams:
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_physics: float = 0.1

    def model_config(self) -> PINNConfig:
        return PINNConfig(hidden_dim=self.hidden_dim, n_layers=self.n_layers, dropout=self.dropout)


@dataclass
class FitResult:
    model: PhysicsInformedNet
    best_val_loss: float
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TrainResult:
    model: PhysicsInformedNet
    best_params: HyperParams
    cv_score: float
    cv_metrics: pd.DataFrame
    checkpoint_path: Path | None = None
    study: optuna.Study | None = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_metrics(model: PhysicsInformedNet, X: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    """R² e MAE por alvo em unidades físicas."""
    pred = predict(model, X)
    metrics: dict[str, float] = {}
    for t in model.target_names:
        metrics[f"{t}_R2"] = float(r2_score(Y[t], pred[t]))
        metrics[f"{t}_MAE"] = float(mean_absolute_error(Y[t], pred[t]))
    return metrics


class PINNTrainer:
    """Orquestra treino, validação cruzada agrupada e ajuste de hiperparâmetros."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        physics: PhysicsWeights | None = None,
        use_collocation: bool = True,
    ) -> None:
        self.config = config or TrainingConfig()
        self.physics = physics or PhysicsWeights()
        self.use_collocation = use_collocation
        self.device = torch.device(self.config.device)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_tr: pd.DataFrame,
        Y_tr: pd.DataFrame,
        params: HyperParams,
        P_tr: pd.DataFrame | None = None,
        X_va: pd.DataFrame | None = None,
        Y_va: pd.DataFrame | None = None,
        P_va: pd.DataFrame | None = None,
        max_epochs: int | None = None,
    ) -> FitResult:
        """Treina um modelo; com validação aplica early stopping."""
        cfg = self.config
        set_seed(cfg.seed)
        stats = NormalizationStats.fit(X_tr, Y_tr)
        model = PhysicsInformedNet(stats, params.model_config()).to(self.device)
        physics = copy.deepcopy(self.physics)
        physics.lambda_physics = params.lambda_physics

        x_tr = torch.tensor(X_tr[stats.feature_names].to_numpy(np.float32), device=self.device)
        y_tr = torch.tensor(stats.transform_Y(Y_tr), device=self.device)
        p_tr = torch.tensor(P_tr.to_numpy(np.float32), device=self.device) if P_tr is not None else None

        has_val = X_va is not None and Y_va is not None
        if has_val:
            x_va = torch.tensor(X_va[stats.feature_names].to_numpy(np.float32), device=self.device)
            y_va = torch.tensor(stats.transform_Y(Y_va), device=self.device)

        opt = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        epochs = max_epochs or cfg.max_epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        gen = torch.Generator().manual_seed(cfg.seed)

        best_state: dict[str, Any] | None = None
        best_val, best_epoch, bad_epochs = float("inf"), epochs - 1, 0
        history: list[dict[str, float]] = []
        n = len(x_tr)

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, generator=gen)
            epoch_logs: list[dict[str, float]] = []
            for i in range(0, n, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size].to(self.device)
                x_col = None
                if self.use_collocation and physics.lambda_physics > 0:
                    x_col = sample_collocation_points(
                        stats.feature_names, max(len(idx) // 2, 8), generator=gen
                    ).to(self.device)
                loss, logs = total_loss(
                    model, x_tr[idx], y_tr[idx], physics,
                    phases_true=None if p_tr is None else p_tr[idx], x_colloc=x_col,
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_logs.append(logs)
            sched.step()

            record = {k: float(np.mean([l[k] for l in epoch_logs])) for k in epoch_logs[0]}
            record["epoch"] = epoch
            if has_val:
                model.eval()
                with torch.no_grad():
                    val = float(torch.mean((model(x_va).properties_norm - y_va) ** 2))
                record["val_mse"] = val
                if val < best_val - 1e-6:
                    best_val, best_epoch, bad_epochs = val, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        history.append(record)
                        break
            history.append(record)

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return FitResult(model=model, best_val_loss=best_val, best_epoch=best_epoch, history=history)

    # ------------------------------------------------------------------ CV
    def cross_validate(
        self,
        dataset: ProcessedDataset,
        params: HyperParams,
        trial: optuna.Trial | None = None,
    ) -> tuple[float, pd.DataFrame, list[int]]:
        """GroupKFold; retorna (MSE normalizado médio, métricas por fold, épocas ótimas)."""
        X, Y, P, groups = dataset.X, dataset.Y, dataset.phases, dataset.groups
        n_splits = min(self.config.n_splits, len(np.unique(groups)))
        gkf = GroupKFold(n_splits=n_splits)
        scores: list[float] = []
        rows: list[dict[str, float]] = []
        epochs: list[int] = []
        for fold, (tr, va) in enumerate(gkf.split(X, Y, groups)):
            res = self.fit(
                X.iloc[tr], Y.iloc[tr], params,
                P_tr=None if P is None else P.iloc[tr],
                X_va=X.iloc[va], Y_va=Y.iloc[va],
            )
            scores.append(res.best_val_loss)
            epochs.append(res.best_epoch + 1)
            rows.append({"fold": fold, "val_mse_norm": res.best_val_loss,
                         **evaluate_metrics(res.model, X.iloc[va], Y.iloc[va])})
            if trial is not None:
                trial.report(float(np.mean(scores)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores)), pd.DataFrame(rows), epochs

    # ------------------------------------------------------------------ Optuna
    def suggest(self, trial: optuna.Trial) -> HyperParams:
        c = self.config
        return HyperParams(
            hidden_dim=trial.suggest_categorical("hidden_dim", list(c.hidden_dim_choices)),
            n_layers=trial.suggest_int("n_layers", *c.n_layers_range),
            dropout=trial.suggest_float("dropout", *c.dropout_range),
            lr=trial.suggest_float("lr", *c.lr_range, log=True),
            weight_decay=trial.suggest_float("weight_decay", *c.weight_decay_range, log=True),
            lambda_physics=trial.suggest_float("lambda_physics", *c.lambda_physics_range, log=True),
        )

    def tune(self, dataset: ProcessedDataset, n_trials: int | None = None) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                    study_name="pinn_al")

        def objective(trial: optuna.Trial) -> float:
            params = self.suggest(trial)
            score, _, epochs = self.cross_validate(dataset, params, trial=trial)
            trial.set_user_attr("mean_best_epoch", float(np.mean(epochs)))
            return score

        study.optimize(objective, n_trials=n_trials or self.config.n_trials)
        return study

    # ------------------------------------------------------------------ pipeline
    def run(
        self,
        dataset: ProcessedDataset,
        tune: bool = True,
        params: HyperParams | None = None,
        save_path: Path | None = DEFAULT_MODEL_PATH,
    ) -> TrainResult:
        """Pipeline completo: (Optuna) -> CV final -> treino em todos os dados."""
        study = None
        if tune:
            study = self.tune(dataset)
            best = HyperParams(**study.best_params)
            logger.info("Melhores hiperparâmetros: %s (CV=%.4f)", best, study.best_value)
        else:
            best = params or HyperParams()

        cv_score, cv_metrics, epochs = self.cross_validate(dataset, best)
        n_epochs = max(int(np.mean(epochs)), 1)
        final = self.fit(dataset.X, dataset.Y, best, P_tr=dataset.phases, max_epochs=n_epochs)

        path = None
        if save_path is not None:
            path = save_checkpoint(final.model, save_path, metadata={
                "hyperparams": asdict(best),
                "cv_score": cv_score,
                "cv_metrics_mean": cv_metrics.mean(numeric_only=True).to_dict(),
                "n_epochs": n_epochs,
                "n_samples": len(dataset.frame),
            })
            logger.info("Modelo salvo em %s", path)
        return TrainResult(model=final.model, best_params=best, cv_score=cv_score,
                           cv_metrics=cv_metrics, checkpoint_path=path, study=study)
