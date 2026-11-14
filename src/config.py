"""Configuração central do sistema de design inverso de ligas AHSS.

Reúne constantes físicas, limites composicionais/processuais, preços de
referência de matérias-primas (LME e mercados de ferroligas), fatores de
emissão de CO2 e hiperparâmetros padrão.

IMPORTANTE: preços e fatores de emissão são *valores indicativos* para
prototipagem. Atualize-os com cotações correntes (LME, Fastmarkets, Argus)
e com o inventário de ciclo de vida (LCA) da sua planta antes de qualquer
decisão econômica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
SYNTHETIC_DIR: Final[Path] = DATA_DIR / "synthetic"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------
R_GAS: Final[float] = 8.314462618  # J/(mol·K)
KELVIN_OFFSET: Final[float] = 273.15
ROOM_TEMPERATURE_C: Final[float] = 25.0
PRESSURE_PA: Final[float] = 101_325.0
# Constante de Koistinen–Marburger (1/°C)
KM_ALPHA: Final[float] = 0.011
# Fração mássica de C na cementita (Fe3C)
CEMENTITE_C_WT: Final[float] = 6.67

# Massas atômicas (g/mol) — usadas na conversão wt% -> fração molar (CALPHAD)
ATOMIC_MASS: Final[dict[str, float]] = {
    "Fe": 55.845, "C": 12.011, "Mn": 54.938, "Si": 28.086, "Cr": 51.996,
    "Ni": 58.693, "Mo": 95.95, "V": 50.942, "Nb": 92.906, "Ti": 47.867,
    "Al": 26.982, "Cu": 63.546, "B": 10.81, "N": 14.007,
}

# ---------------------------------------------------------------------------
# Espaço de projeto
# ---------------------------------------------------------------------------
# Elementos de liga (Fe = balanço). Ordem define a ordem das features.
ELEMENTS: Final[tuple[str, ...]] = (
    "C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Nb", "Ti", "Al", "Cu", "B", "N",
)

# Limites composicionais (wt%) típicos de AHSS (DP, TRIP, CP, Q&P, MS)
COMPOSITION_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "C": (0.05, 0.40),
    "Mn": (0.50, 3.50),
    "Si": (0.00, 2.00),
    "Cr": (0.00, 1.00),
    "Ni": (0.00, 1.00),
    "Mo": (0.00, 0.50),
    "V": (0.00, 0.15),
    "Nb": (0.00, 0.08),
    "Ti": (0.00, 0.08),
    "Al": (0.00, 1.00),
    "Cu": (0.00, 0.50),
    "B": (0.00, 0.004),
    "N": (0.002, 0.010),
}

# Variáveis de processo (recozimento intercrítico + resfriamento)
PROCESS_FEATURES: Final[tuple[str, ...]] = (
    "T_anneal_C", "cooling_rate_Cps", "grain_size_um",
)
PROCESS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "T_anneal_C": (720.0, 950.0),
    "cooling_rate_Cps": (1.0, 100.0),
    "grain_size_um": (2.0, 20.0),
}
# Features amostradas/normalizadas em escala log10
LOG_FEATURES: Final[frozenset[str]] = frozenset({"cooling_rate_Cps"})

INPUT_FEATURES: Final[tuple[str, ...]] = ELEMENTS + PROCESS_FEATURES

# Saídas do modelo
TARGETS: Final[tuple[str, ...]] = ("YS_MPa", "UTS_MPa", "El_pct")
PHASES: Final[tuple[str, ...]] = ("f_ferrite", "f_bainite", "f_martensite", "f_retained_austenite")

# Restrições de monotonicidade físicas: (alvo, feature) -> sinal da derivada
#   +1: alvo deve crescer com a feature; -1: deve decrescer.
MONOTONIC_CONSTRAINTS: Final[dict[tuple[str, str], int]] = {
    ("UTS_MPa", "C"): +1,                 # endurecimento por solução sólida/martensita
    ("UTS_MPa", "Mn"): +1,                # solução sólida + temperabilidade
    ("UTS_MPa", "cooling_rate_Cps"): +1,  # mais martensita com resfriamento rápido
    ("YS_MPa", "grain_size_um"): -1,      # Hall–Petch
}

# ---------------------------------------------------------------------------
# Economia e sustentabilidade
# ---------------------------------------------------------------------------
# Preço de referência (USD/kg de elemento contido). Fe = sucata/gusa.
# Ni, Cu, Al, Mo: referências LME; demais: mercados de ferroligas.
ELEMENT_PRICE_USD_PER_KG: Final[dict[str, float]] = {
    "Fe": 0.42, "C": 0.35, "Mn": 1.90, "Si": 2.10, "Cr": 2.60, "Ni": 16.50,
    "Mo": 44.00, "V": 32.00, "Nb": 45.00, "Ti": 6.50, "Al": 2.55, "Cu": 9.40,
    "B": 5.00, "N": 1.00,
}

# Fator de emissão "cradle-to-gate" (kg CO2e / kg de elemento).
# Base: Nuss & Eckelman (2014), PLoS ONE 9(7): e101298 — valores arredondados.
# Fe: rota BF-BOF (~1.9); para rota EAF use FE_CO2_EAF.
ELEMENT_CO2_KG_PER_KG: Final[dict[str, float]] = {
    "Fe": 1.90, "C": 2.00, "Mn": 1.00, "Si": 5.00, "Cr": 2.40, "Ni": 6.50,
    "Mo": 5.70, "V": 33.00, "Nb": 12.50, "Ti": 8.10, "Al": 8.20, "Cu": 2.80,
    "B": 1.50, "N": 0.50,
}
FE_CO2_EAF: Final[float] = 0.60

# Rendimento metálico de adição (recuperação) — perdas por oxidação na aciaria
ELEMENT_RECOVERY: Final[dict[str, float]] = {
    "C": 0.90, "Mn": 0.92, "Si": 0.85, "Cr": 0.95, "Ni": 1.00, "Mo": 1.00,
    "V": 0.90, "Nb": 0.90, "Ti": 0.60, "Al": 0.50, "Cu": 1.00, "B": 0.50, "N": 0.70,
}

# ---------------------------------------------------------------------------
# Limites metalúrgicos (filtro termodinâmico / risco)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetallurgicalLimits:
    """Limites usados pelo filtro termodinâmico e de risco."""

    ceq_iiw_max: float = 0.60          # soldabilidade (IIW) — AHSS automotivo
    pcm_max: float = 0.30              # Ito–Bessyo, risco de trinca a frio
    min_intercritical_window_C: float = 30.0  # Ac3 - Ac1 mínimo
    max_mn_banding: float = 3.0        # bandeamento/segregação de Mn
    max_si_galvanizing: float = 1.5    # óxidos de Si prejudicam galvanização
    max_al_casting: float = 1.0        # entupimento de válvula / AlN
    max_b_embrittlement: float = 0.005
    max_microalloy_sum: float = 0.15   # Nb+Ti+V: carbonitretos grosseiros
    cu_ni_ratio_hot_shortness: float = 2.0  # Cu/Ni > 2 -> fragilidade a quente
    min_ti_n_ratio: float = 3.42       # estequiometria TiN para proteger o B
    max_undesired_phase_fraction: float = 0.01  # sigma, Laves, etc. (CALPHAD)


# ---------------------------------------------------------------------------
# Hiperparâmetros
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuração padrão de treinamento da PINN."""

    n_splits: int = 5
    n_trials: int = 30
    max_epochs: int = 300
    patience: int = 30
    batch_size: int = 256
    seed: int = 42
    device: str = "cpu"
    # Espaço de busca do Optuna
    hidden_dim_choices: tuple[int, ...] = (64, 128, 256)
    n_layers_range: tuple[int, int] = (2, 5)
    lr_range: tuple[float, float] = (1e-4, 3e-3)
    weight_decay_range: tuple[float, float] = (1e-7, 1e-3)
    dropout_range: tuple[float, float] = (0.0, 0.2)
    lambda_physics_range: tuple[float, float] = (1e-3, 1.0)


@dataclass
class OptimizationConfig:
    """Configuração padrão do design inverso (NSGA-III)."""

    pop_size: int = 120
    n_generations: int = 150
    n_partitions: int = 6
    seed: int = 1
    min_uts_MPa: float = 980.0
    min_elongation_pct: float = 12.0
    ceq_max: float = 0.60
    max_risk_score: float = 1.0
    max_uts_cv: float | None = None   # coef. de variação máx. (MC-dropout)
    n_mc_samples: int = 0             # 0 = predição determinística
    fe_route: str = "BF-BOF"          # ou "EAF"
    fixed_values: dict[str, float] = field(default_factory=dict)
