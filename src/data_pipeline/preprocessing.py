"""Limpeza, engenharia de features metalúrgicas, outliers e escalonamento."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import metallurgy as met
from src.config import (
    ELEMENTS,
    INPUT_FEATURES,
    LOG_FEATURES,
    PHASES,
    PROCESS_FEATURES,
    PROCESSED_DIR,
    TARGETS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Features metalúrgicas
# ---------------------------------------------------------------------------
def compute_carbon_equivalent(df: pd.DataFrame, method: str = "IIW") -> pd.Series:
    """Carbono equivalente (``IIW`` ou ``Pcm``)."""
    comp = {el: df[el] for el in ELEMENTS if el in df}
    if method.upper() == "IIW":
        return pd.Series(met.carbon_equivalent_iiw(comp), index=df.index, name="Ceq_IIW")
    if method.upper() == "PCM":
        return pd.Series(met.pcm_ito_bessyo(comp), index=df.index, name="Pcm")
    raise ValueError(f"Método de C_eq desconhecido: {method}")


def add_metallurgical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona C_eq, Pcm, temperaturas críticas e equivalentes de Schaeffler."""
    out = df.copy()
    comp = {el: out[el] for el in ELEMENTS if el in out}
    out["Fe_balance"] = 100.0 - out[[el for el in ELEMENTS if el in out]].sum(axis=1)
    out["Ceq_IIW"] = compute_carbon_equivalent(out, "IIW")
    out["Pcm"] = compute_carbon_equivalent(out, "Pcm")
    out["Ac1_C"] = met.ac1_andrews(comp)
    out["Ac3_C"] = met.ac3_andrews(comp)
    out["intercritical_window_C"] = out["Ac3_C"] - out["Ac1_C"]
    out["Ms_C"] = met.ms_andrews(comp)
    out["Bs_C"] = met.bs_steven_haynes(comp)
    cr_eq, ni_eq = met.schaeffler_equivalents(comp)
    out["Cr_eq"], out["Ni_eq"] = cr_eq, ni_eq
    return out


# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Contabiliza as linhas removidas em cada etapa."""

    n_input: int = 0
    n_missing_required: int = 0
    n_invalid_composition: int = 0
    n_physical_violations: int = 0
    n_duplicates: int = 0
    n_outliers: int = 0
    n_output: int = 0

    def __str__(self) -> str:
        return " | ".join(f"{k}={v}" for k, v in asdict(self).items())


def clean_dataset(
    df: pd.DataFrame,
    targets: Sequence[str] = TARGETS,
    max_alloy_sum: float = 15.0,
    report: CleaningReport | None = None,
) -> pd.DataFrame:
    """Limpeza básica de dados experimentais ou sintéticos.

    * Elementos não reportados -> 0 wt% (convenção de certificados de corrida).
    * Remove linhas sem processo/alvos, composições negativas ou com soma de
      liga acima de ``max_alloy_sum``.
    * Remove violações físicas: YS > UTS, El <= 0, UTS <= 0.
    * Remove duplicatas exatas.
    """
    report = report if report is not None else CleaningReport()
    report.n_input = len(df)
    out = df.copy()

    for el in ELEMENTS:
        if el not in out:
            out[el] = 0.0
    out[list(ELEMENTS)] = out[list(ELEMENTS)].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    required = list(PROCESS_FEATURES) + [t for t in targets if t in out]
    n0 = len(out)
    out = out.dropna(subset=[c for c in required if c in out])
    report.n_missing_required = n0 - len(out)

    n0 = len(out)
    comp = out[list(ELEMENTS)]
    valid = (comp >= 0).all(axis=1) & (comp.sum(axis=1) <= max_alloy_sum)
    out = out[valid]
    report.n_invalid_composition = n0 - len(out)

    n0 = len(out)
    phys = pd.Series(True, index=out.index)
    if {"YS_MPa", "UTS_MPa"} <= set(out):
        phys &= out["YS_MPa"] <= out["UTS_MPa"]
    if "UTS_MPa" in out:
        phys &= out["UTS_MPa"] > 0
    if "El_pct" in out:
        phys &= out["El_pct"] > 0
    if "cooling_rate_Cps" in out:
        phys &= out["cooling_rate_Cps"] > 0
    out = out[phys]
    report.n_physical_violations = n0 - len(out)

    n0 = len(out)
    out = out.drop_duplicates(subset=[c for c in INPUT_FEATURES + tuple(targets) if c in out])
    report.n_duplicates = n0 - len(out)

    report.n_output = len(out)
    return out.reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    contamination: float | str = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Remove outliers multivariados com IsolationForest.

    Returns
    -------
    (df_limpo, máscara_inliers)
    """
    cols = list(columns) if columns is not None else [
        c for c in (*INPUT_FEATURES, *TARGETS) if c in df
    ]
    if len(df) < 10:
        return df.copy(), np.ones(len(df), dtype=bool)
    X = df[cols].to_numpy(float)
    X = X.copy()
    for j, c in enumerate(cols):
        if c in LOG_FEATURES:
            X[:, j] = np.log10(np.clip(X[:, j], 1e-6, None))
    forest = IsolationForest(contamination=contamination, random_state=seed, n_estimators=300)
    mask = forest.fit_predict(X) == 1
    return df.loc[mask].reset_index(drop=True), mask


# ---------------------------------------------------------------------------
# Escalonamento
# ---------------------------------------------------------------------------
@dataclass
class NormalizationStats:
    """Estatísticas de normalização (z-score, com log10 opcional por feature).

    São embutidas como *buffers* na PINN, de modo que o modelo recebe valores
    físicos brutos e as derivadas via autograd ficam em unidades físicas.
    """

    feature_names: list[str]
    target_names: list[str]
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]
    log_features: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        log_features: Iterable[str] = LOG_FEATURES,
    ) -> "NormalizationStats":
        logs = [c for c in X.columns if c in set(log_features)]
        Xt = X.astype(float).copy()
        for c in logs:
            Xt[c] = np.log10(np.clip(Xt[c], 1e-6, None))
        x_std = Xt.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        y_std = Y.astype(float).std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        return cls(
            feature_names=list(X.columns),
            target_names=list(Y.columns),
            x_mean=Xt.mean().tolist(),
            x_std=x_std.tolist(),
            y_mean=Y.astype(float).mean().tolist(),
            y_std=y_std.tolist(),
            log_features=logs,
        )

    def transform_X(self, X: pd.DataFrame) -> np.ndarray:
        Xt = X[self.feature_names].astype(float).copy()
        for c in self.log_features:
            Xt[c] = np.log10(np.clip(Xt[c], 1e-6, None))
        return ((Xt.to_numpy() - np.asarray(self.x_mean)) / np.asarray(self.x_std)).astype(np.float32)

    def transform_Y(self, Y: pd.DataFrame) -> np.ndarray:
        Yt = Y[self.target_names].astype(float).to_numpy()
        return ((Yt - np.asarray(self.y_mean)) / np.asarray(self.y_std)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        return cls(**json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class ProcessedDataset:
    """Dataset pronto para o treinamento."""

    frame: pd.DataFrame
    stats: NormalizationStats
    report: CleaningReport

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[list(INPUT_FEATURES)]

    @property
    def Y(self) -> pd.DataFrame:
        return self.frame[list(TARGETS)]

    @property
    def phases(self) -> pd.DataFrame | None:
        cols = [p for p in PHASES if p in self.frame]
        return self.frame[cols] if len(cols) == len(PHASES) else None

    @property
    def groups(self) -> np.ndarray:
        if "alloy_id" in self.frame:
            return self.frame["alloy_id"].to_numpy()
        # Agrupa por composição idêntica (mesma corrida, processos distintos)
        return self.frame.groupby(list(ELEMENTS), sort=False).ngroup().to_numpy()


def preprocess(
    df: pd.DataFrame,
    contamination: float | str = 0.02,
    seed: int = 42,
    save_dir: Path | None = PROCESSED_DIR,
    name: str = "dataset",
) -> ProcessedDataset:
    """Executa limpeza -> outliers -> features metalúrgicas -> estatísticas."""
    report = CleaningReport()
    clean = clean_dataset(df, report=report)
    n0 = len(clean)
    clean, _ = remove_outliers(clean, contamination=contamination, seed=seed)
    report.n_outliers = n0 - len(clean)
    report.n_output = len(clean)
    clean = add_metallurgical_features(clean)

    stats = NormalizationStats.fit(clean[list(INPUT_FEATURES)], clean[list(TARGETS)])
    logger.info("Pré-processamento: %s", report)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        clean.to_csv(save_dir / f"{name}.csv", index=False)
        stats.save(save_dir / f"{name}_normalization.json")
    return ProcessedDataset(frame=clean, stats=stats, report=report)
