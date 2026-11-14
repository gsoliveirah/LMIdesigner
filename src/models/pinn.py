"""Rede Neural Físico-Informada (PINN) para propriedades de aços AHSS.

Arquitetura
-----------
Entrada física bruta (wt%, °C, °C/s, µm)
  -> normalização interna (buffers; log10 opcional)
  -> tronco MLP residual (SiLU + LayerNorm + Dropout)
  -> cabeça mecânica  : [YS, UTS, El]
  -> cabeça de fases  : softmax[ferrita, bainita, martensita, austenita retida]

Como a normalização está *dentro* do modelo, ``autograd`` fornece derivadas
em unidades físicas e o otimizador inverso pode consultar o modelo
diretamente com composições reais.

Termos físicos da função de perda
---------------------------------
1. ``ys_le_uts``   — desigualdade YS ≤ UTS.
2. ``rule_of_mixtures`` — UTS/YS consistentes com as frações de fase
   previstas e a resistência dos constituintes (balanço de massa de C).
3. ``elongation_law`` — compromisso resistência-ductilidade + efeito TRIP.
4. ``monotonicity`` — sinais de derivadas via autograd (ex.: ∂UTS/∂C ≥ 0,
   Hall–Petch ∂YS/∂d ≤ 0).
5. ``non_negative`` — El ≥ 0.
A conservação de fases (Σf = 1, f ≥ 0) é imposta *rigidamente* pelo softmax.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src import metallurgy as met
from src.config import (
    COMPOSITION_BOUNDS,
    ELEMENTS,
    LOG_FEATURES,
    MONOTONIC_CONSTRAINTS,
    PHASES,
    PROCESS_BOUNDS,
)
from src.data_pipeline.preprocessing import NormalizationStats


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
@dataclass
class PINNConfig:
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.05
    n_phases: int = len(PHASES)


@dataclass
class PhysicsWeights:
    """Pesos relativos dos termos físicos (multiplicados por ``lambda_physics``)."""

    lambda_physics: float = 0.1
    ys_le_uts: float = 1.0
    rule_of_mixtures: float = 1.0
    elongation_law: float = 0.5
    monotonicity: float = 1.0
    non_negative: float = 1.0
    phase_supervision: float = 1.0
    monotonic_constraints: dict[tuple[str, str], int] = field(
        default_factory=lambda: dict(MONOTONIC_CONSTRAINTS)
    )


@dataclass
class PINNOutput:
    properties: torch.Tensor        # (N, n_targets) em unidades físicas
    properties_norm: torch.Tensor   # (N, n_targets) normalizado
    phases: torch.Tensor            # (N, n_phases), soma = 1


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc1(self.norm(x)))
        h = self.drop(h)
        return x + self.fc2(h)


class PhysicsInformedNet(nn.Module):
    """PINN multitarefa (propriedades + microestrutura)."""

    def __init__(self, stats: NormalizationStats, config: PINNConfig | None = None) -> None:
        super().__init__()
        self.config = config or PINNConfig()
        self.stats = stats
        self.feature_names: list[str] = list(stats.feature_names)
        self.target_names: list[str] = list(stats.target_names)
        self.phase_names: list[str] = list(PHASES)

        f32 = torch.float32
        self.register_buffer("x_mean", torch.tensor(stats.x_mean, dtype=f32))
        self.register_buffer("x_std", torch.tensor(stats.x_std, dtype=f32))
        self.register_buffer("y_mean", torch.tensor(stats.y_mean, dtype=f32))
        self.register_buffer("y_std", torch.tensor(stats.y_std, dtype=f32))
        log_mask = [name in set(stats.log_features) for name in self.feature_names]
        self.register_buffer("log_mask", torch.tensor(log_mask, dtype=torch.bool))

        cfg = self.config
        n_in = len(self.feature_names)
        self.input_layer = nn.Linear(n_in, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.head_properties = nn.Linear(cfg.hidden_dim, len(self.target_names))
        self.head_phases = nn.Linear(cfg.hidden_dim, cfg.n_phases)

    # -- utilidades -----------------------------------------------------------
    def normalize_inputs(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = torch.where(self.log_mask, torch.log10(torch.clamp(x_raw, min=1e-6)), x_raw)
        return (x - self.x_mean) / self.x_std

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    def target_index(self, name: str) -> int:
        return self.target_names.index(name)

    # -- forward ----------------------------------------------------------------
    def forward(self, x_raw: torch.Tensor) -> PINNOutput:
        h = F.silu(self.input_layer(self.normalize_inputs(x_raw)))
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        y_norm = self.head_properties(h)
        phases = torch.softmax(self.head_phases(h), dim=-1)
        y = y_norm * self.y_std + self.y_mean
        return PINNOutput(properties=y, properties_norm=y_norm, phases=phases)


# ---------------------------------------------------------------------------
# Perdas
# ---------------------------------------------------------------------------
def _composition_from_tensor(model: PhysicsInformedNet, x_raw: torch.Tensor) -> dict[str, torch.Tensor]:
    return {el: x_raw[:, model.feature_index(el)] for el in ELEMENTS if el in model.feature_names}


def physics_residuals(
    model: PhysicsInformedNet,
    x_raw: torch.Tensor,
    weights: PhysicsWeights,
) -> dict[str, torch.Tensor]:
    """Calcula os resíduos físicos em ``x_raw`` (dados ou pontos de colocação).

    ``x_raw`` deve ter ``requires_grad=True`` para o termo de monotonicidade.
    """
    out = model(x_raw)
    y, phases = out.properties, out.phases
    ys = y[:, model.target_index("YS_MPa")]
    uts = y[:, model.target_index("UTS_MPa")]
    el = y[:, model.target_index("El_pct")]
    s_ys = model.y_std[model.target_index("YS_MPa")]
    s_uts = model.y_std[model.target_index("UTS_MPa")]
    s_el = model.y_std[model.target_index("El_pct")]

    terms: dict[str, torch.Tensor] = {}
    terms["ys_le_uts"] = torch.mean(F.relu(ys - uts) / s_uts) ** 2

    comp = _composition_from_tensor(model, x_raw)
    phase_map = {name: phases[:, i] for i, name in enumerate(model.phase_names)}
    d = x_raw[:, model.feature_index("grain_size_um")]
    ys_rom, uts_rom = met.rule_of_mixtures(comp, phase_map, d)
    terms["rule_of_mixtures"] = torch.mean(((uts - uts_rom) / s_uts) ** 2) + torch.mean(
        ((ys - ys_rom) / s_ys) ** 2
    )

    el_law = met.elongation_model(uts, phase_map["f_retained_austenite"])
    terms["elongation_law"] = torch.mean(((el - el_law) / s_el) ** 2)
    terms["non_negative"] = torch.mean((F.relu(-el) / s_el) ** 2)

    mono = torch.zeros((), device=x_raw.device)
    if x_raw.requires_grad and weights.monotonic_constraints:
        for (target, feature), sign in weights.monotonic_constraints.items():
            if target not in model.target_names or feature not in model.feature_names:
                continue
            t_idx = model.target_index(target)
            f_idx = model.feature_index(feature)
            grad = torch.autograd.grad(
                y[:, t_idx].sum(), x_raw, create_graph=True, retain_graph=True
            )[0][:, f_idx]
            scale = x_raw[:, f_idx].detach().std().clamp(min=1e-6) / model.y_std[t_idx]
            mono = mono + torch.mean(F.relu(-sign * grad * scale) ** 2)
    terms["monotonicity"] = mono
    return terms


def total_loss(
    model: PhysicsInformedNet,
    x_raw: torch.Tensor,
    y_norm_true: torch.Tensor,
    weights: PhysicsWeights,
    phases_true: torch.Tensor | None = None,
    x_colloc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perda total = dados + supervisão de fases + λ·Σ wᵢ·resíduoᵢ."""
    out = model(x_raw)
    data_loss = F.mse_loss(out.properties_norm, y_norm_true)
    loss = data_loss
    logs: dict[str, float] = {"data": float(data_loss.detach())}

    if phases_true is not None:
        phase_loss = F.mse_loss(out.phases, phases_true) * 10.0
        loss = loss + weights.phase_supervision * phase_loss
        logs["phases"] = float(phase_loss.detach())

    if weights.lambda_physics > 0:
        x_phys = x_raw if x_colloc is None else torch.cat([x_raw, x_colloc], dim=0)
        x_phys = x_phys.detach().requires_grad_(True)
        terms = physics_residuals(model, x_phys, weights)
        phys = sum(getattr(weights, k) * v for k, v in terms.items())
        loss = loss + weights.lambda_physics * phys
        logs.update({f"phys_{k}": float(v.detach()) for k, v in terms.items()})
    logs["total"] = float(loss.detach())
    return loss, logs


# ---------------------------------------------------------------------------
# Pontos de colocação
# ---------------------------------------------------------------------------
def sample_collocation_points(
    feature_names: list[str],
    n: int,
    generator: torch.Generator | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Amostra pontos uniformes no espaço de projeto (sem rótulos).

    Os resíduos físicos são impostos também fora dos dados, regularizando o
    modelo nas regiões que o otimizador inverso irá explorar.
    """
    bounds = dict(bounds) if bounds is not None else {**COMPOSITION_BOUNDS, **PROCESS_BOUNDS}
    u = torch.rand((n, len(feature_names)), generator=generator)
    cols = []
    for j, name in enumerate(feature_names):
        lo, hi = bounds[name]
        if name in LOG_FEATURES:
            cols.append(10 ** (np.log10(lo) + u[:, j] * (np.log10(hi) - np.log10(lo))))
        else:
            cols.append(lo + u[:, j] * (hi - lo))
    return torch.stack(cols, dim=1).float()


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------
def _enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict(
    model: PhysicsInformedNet,
    X: pd.DataFrame | np.ndarray,
    n_mc_samples: int = 0,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Predição (opcionalmente com incerteza por MC-dropout).

    Retorna colunas ``<alvo>``, ``<fase>`` e, se ``n_mc_samples > 0``,
    ``<alvo>_std``.
    """
    arr = X[model.feature_names].to_numpy(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, np.float32)
    device = next(model.parameters()).device
    x = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).to(device)

    def _run() -> tuple[np.ndarray, np.ndarray]:
        props, phs = [], []
        for i in range(0, len(x), batch_size):
            o = model(x[i:i + batch_size])
            props.append(o.properties.cpu().numpy())
            phs.append(o.phases.cpu().numpy())
        return np.concatenate(props), np.concatenate(phs)

    was_training = model.training
    if n_mc_samples > 0:
        _enable_mc_dropout(model)
        samples = [_run() for _ in range(n_mc_samples)]
        props = np.stack([s[0] for s in samples])
        phases = np.stack([s[1] for s in samples]).mean(0)
        mean, std = props.mean(0), props.std(0)
    else:
        model.eval()
        mean, phases = _run()
        std = None
    model.train(was_training)

    index = X.index if isinstance(X, pd.DataFrame) else None
    df = pd.DataFrame(mean, columns=model.target_names, index=index)
    for i, p in enumerate(model.phase_names):
        df[p] = phases[:, i]
    if std is not None:
        for i, t in enumerate(model.target_names):
            df[f"{t}_std"] = std[:, i]
    return df


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def save_checkpoint(model: PhysicsInformedNet, path: str | Path, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "stats": asdict(model.stats),
            "metadata": metadata or {},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[PhysicsInformedNet, dict]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model = PhysicsInformedNet(NormalizationStats(**ckpt["stats"]), PINNConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("metadata", {})
