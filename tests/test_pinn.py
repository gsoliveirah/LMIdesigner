from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.config import TrainingConfig
from src.models.pinn import (
    PhysicsInformedNet,
    PhysicsWeights,
    load_checkpoint,
    physics_residuals,
    predict,
    sample_collocation_points,
    save_checkpoint,
    total_loss,
)
from src.models.train import HyperParams, PINNTrainer


def test_forward_shapes_and_phase_simplex(dataset) -> None:
    model = PhysicsInformedNet(dataset.stats)
    x = torch.tensor(dataset.X.to_numpy(np.float32)[:16])
    out = model(x)
    assert out.properties.shape == (16, 3)
    assert out.phases.shape == (16, 4)
    torch.testing.assert_close(out.phases.sum(-1), torch.ones(16))
    assert (out.phases >= 0).all()


def test_physics_residuals_finite_and_differentiable(dataset) -> None:
    model = PhysicsInformedNet(dataset.stats)
    x = sample_collocation_points(model.feature_names, 64).requires_grad_(True)
    terms = physics_residuals(model, x, PhysicsWeights())
    assert set(terms) == {"ys_le_uts", "rule_of_mixtures", "elongation_law", "non_negative", "monotonicity"}
    total = sum(terms.values())
    assert torch.isfinite(total)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_monotonicity_penalty_detects_violation(dataset) -> None:
    """Um modelo com ∂UTS/∂C < 0 deve ser penalizado; com ∂UTS/∂C > 0, não."""
    from src.models.pinn import PINNOutput

    model = PhysicsInformedNet(dataset.stats)
    c_i = model.feature_index("C")

    def fake_forward(x: torch.Tensor) -> PINNOutput:
        c = x[:, c_i]
        props = torch.stack([400 + 0 * c, 1000 - 500 * c, 20 + 0 * c], dim=1)  # UTS decresce com C
        return PINNOutput(properties=props, properties_norm=(props - model.y_mean) / model.y_std,
                          phases=torch.full((len(x), 4), 0.25))

    model.forward = fake_forward  # type: ignore[method-assign]
    x = sample_collocation_points(model.feature_names, 64).requires_grad_(True)
    bad = physics_residuals(model, x, PhysicsWeights(monotonic_constraints={("UTS_MPa", "C"): +1}))
    assert bad["monotonicity"] > 0
    x2 = sample_collocation_points(model.feature_names, 64).requires_grad_(True)
    ok = physics_residuals(model, x2, PhysicsWeights(monotonic_constraints={("UTS_MPa", "C"): -1}))
    assert ok["monotonicity"] == 0


def test_training_reduces_loss_and_learns(dataset, trained_model) -> None:
    pred = predict(trained_model, dataset.X)
    corr = np.corrcoef(pred["UTS_MPa"], dataset.Y["UTS_MPa"])[0, 1]
    assert corr > 0.8
    np.testing.assert_allclose(pred[["f_ferrite", "f_bainite", "f_martensite",
                                     "f_retained_austenite"]].sum(axis=1), 1.0, atol=1e-5)


def test_total_loss_components(dataset) -> None:
    model = PhysicsInformedNet(dataset.stats)
    x = torch.tensor(dataset.X.to_numpy(np.float32)[:32])
    y = torch.tensor(dataset.stats.transform_Y(dataset.Y.iloc[:32]))
    p = torch.tensor(dataset.phases.to_numpy(np.float32)[:32])
    loss, logs = total_loss(model, x, y, PhysicsWeights(), phases_true=p,
                            x_colloc=sample_collocation_points(model.feature_names, 16))
    assert torch.isfinite(loss)
    assert {"data", "phases", "phys_rule_of_mixtures", "total"} <= set(logs)


def test_mc_dropout_uncertainty(dataset, trained_model) -> None:
    model = PhysicsInformedNet(dataset.stats)  # dropout > 0 por padrão
    pred = predict(model, dataset.X.head(10), n_mc_samples=8)
    assert "UTS_MPa_std" in pred and (pred["UTS_MPa_std"] >= 0).all()


def test_checkpoint_roundtrip(tmp_path, dataset, trained_model) -> None:
    path = save_checkpoint(trained_model, tmp_path / "m.pt", metadata={"a": 1})
    loaded, meta = load_checkpoint(path)
    assert meta == {"a": 1}
    pd.testing.assert_frame_equal(predict(trained_model, dataset.X.head(5)),
                                  predict(loaded, dataset.X.head(5)))


def test_group_kfold_and_optuna_smoke(dataset) -> None:
    trainer = PINNTrainer(TrainingConfig(n_splits=2, n_trials=2, max_epochs=3, patience=3,
                                         hidden_dim_choices=(32,), n_layers_range=(1, 2)))
    score, metrics, epochs = trainer.cross_validate(dataset, HyperParams(hidden_dim=32, n_layers=1))
    assert np.isfinite(score) and len(metrics) == 2 and len(epochs) == 2
    study = trainer.tune(dataset, n_trials=2)
    assert len(study.trials) == 2


def test_group_kfold_has_no_alloy_leakage(dataset) -> None:
    from sklearn.model_selection import GroupKFold

    groups = dataset.groups
    for tr, va in GroupKFold(n_splits=3).split(dataset.X, dataset.Y, groups):
        assert not set(groups[tr]) & set(groups[va])
