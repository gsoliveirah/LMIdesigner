"""Configuração central do sistema de design inverso de ligas de alumínio secundário.

Domínio-alvo: ligas de fundição Al-Si-(Mg/Cu) produzidas a partir de sucata
(família A356/A357, 319/AlSi6Cu4, A380/AlSi9Cu3(Fe), AlSi10Mg) com
tratamento térmico de solubilização + envelhecimento artificial (T6/T7).

O módulo reúne:

* constantes físicas e estequiometria das fases intermetálicas;
* limites do espaço de projeto (composição e processo);
* preços de referência (LME e mercado de sucata/ligas-mãe) e fatores de
  emissão de CO2 *cradle-to-gate*;
* composição típica de classes de sucata e do alumínio primário (diluente);
* limites metalúrgicos e hiperparâmetros padrão.

IMPORTANTE: preços e fatores de emissão são *valores indicativos* para
prototipagem. Atualize-os com cotações correntes (LME, Fastmarkets, Argus,
Platts) e com o inventário de ciclo de vida (LCA) da sua fundição antes de
qualquer decisão econômica.
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

DEFAULT_DATASET_NAME: Final[str] = "synthetic_al.csv"
DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "pinn_al.pt"

# ---------------------------------------------------------------------------
# Constantes físicas
# ---------------------------------------------------------------------------
R_GAS: Final[float] = 8.314462618  # J/(mol·K)
KELVIN_OFFSET: Final[float] = 273.15
PRESSURE_PA: Final[float] = 101_325.0

AL_MELTING_C: Final[float] = 660.45
AL_SI_EUTECTIC_C: Final[float] = 577.0
AL_SI_EUTECTIC_SI_WT: Final[float] = 12.6
# Eutéticos de baixo ponto de fusão (fusão incipiente na solubilização)
AL_SI_CU_EUTECTIC_C: Final[float] = 525.0      # Al + Si + Al2Cu
AL_SI_CU_MG_EUTECTIC_C: Final[float] = 507.0   # Al + Si + Al2Cu + Q-Al5Cu2Mg8Si6

# Massas atômicas (g/mol) — conversão wt% <-> fração molar (CALPHAD)
ATOMIC_MASS: Final[dict[str, float]] = {
    "Al": 26.982, "Si": 28.086, "Fe": 55.845, "Cu": 63.546, "Mn": 54.938,
    "Mg": 24.305, "Zn": 65.38, "Cr": 51.996, "Ni": 58.693, "Ti": 47.867, "Sr": 87.62,
}

# Fração mássica dos elementos nas fases intermetálicas (estequiometria)
#   β-Al5FeSi        : Fe 25.5 %, Si 12.8 %
#   α-Al15(Fe,Mn)3Si2: (Fe+Mn) 26.5 %, Si 9.0 %   ("escrita chinesa")
#   θ-Al2Cu          : Cu 54.1 %
#   β-Mg2Si          : Mg 63.4 %, Si 36.6 %
BETA_FE_WT: Final[float] = 0.255
BETA_SI_WT: Final[float] = 0.128
ALPHA_FEMN_WT: Final[float] = 0.265
ALPHA_SI_WT: Final[float] = 0.090
AL2CU_CU_WT: Final[float] = 0.541
MG2SI_SI_PER_MG: Final[float] = 28.086 / (2 * 24.305)  # 0.578 kg Si / kg Mg

# Solubilidade máxima no Al-α (wt%) e entalpias aparentes de dissolução (J/mol)
CU_MAX_SOLUBILITY_WT: Final[float] = 5.65   # a 548 °C (binário Al-Cu)
CU_SOLVUS_REF_C: Final[float] = 548.0
CU_DISSOLUTION_H: Final[float] = 40_000.0
MG_MAX_SOLUBILITY_WT: Final[float] = 0.70   # Mg em equilíbrio com Si a ~555 °C
MG_SOLVUS_REF_C: Final[float] = 555.0
MG_DISSOLUTION_H: Final[float] = 55_000.0
SI_IN_MATRIX_WT: Final[float] = 0.5         # Si retido no Al-α após T6

# Cinética de envelhecimento (Shercliff–Ashby): energia de ativação e pico de referência
AGING_ACTIVATION_J: Final[float] = 130_000.0
AGING_REF_C: Final[float] = 170.0
AGING_PEAK_TIME_REF_H: Final[float] = 6.0

# Espaçamento dendrítico secundário: SDAS = A · CR^(-n)  (A356, µm e °C/s)
SDAS_A_UM: Final[float] = 39.4
SDAS_N: Final[float] = 0.317

# ---------------------------------------------------------------------------
# Espaço de projeto
# ---------------------------------------------------------------------------
# Elementos de liga/impurezas (Al = balanço). Ordem define a ordem das features.
ELEMENTS: Final[tuple[str, ...]] = ("Si", "Fe", "Cu", "Mn", "Mg", "Zn", "Cr", "Ni", "Ti", "Sr")

# Impurezas "acumulativas" da reciclagem: não podem ser removidas por adição,
# só por diluição com alumínio primário (ou refino por sludging, no caso do Fe).
TRAMP_ELEMENTS: Final[frozenset[str]] = frozenset({"Fe", "Cu", "Zn", "Ni", "Cr"})
# Elementos que nunca são adicionados deliberadamente ao banho: seu teor final é
# consequência da carga (sucata × primário). O valor-alvo atua como teto (spec máx.).
# São as impurezas acumulativas, exceto o Cu (adicionado deliberadamente em 319/A380).
# Fe: em injeção sob pressão pode-se adicionar AlFe contra a soldagem na matriz —
# nesse caso remova-o deste conjunto.
NON_ADDABLE_ELEMENTS: Final[frozenset[str]] = TRAMP_ELEMENTS - {"Cu"}

# Limites composicionais (wt%) — ligas secundárias hipoeutéticas Al-Si
COMPOSITION_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "Si": (5.0, 11.5),
    "Fe": (0.10, 1.30),
    "Cu": (0.00, 4.00),
    "Mn": (0.00, 0.80),
    "Mg": (0.05, 0.65),
    "Zn": (0.00, 2.00),
    "Cr": (0.00, 0.20),
    "Ni": (0.00, 0.30),
    "Ti": (0.00, 0.20),
    "Sr": (0.00, 0.04),
}

# Variáveis de processo: solidificação + T6
PROCESS_FEATURES: Final[tuple[str, ...]] = (
    "cooling_rate_Cps", "T_solution_C", "T_aging_C", "t_aging_h",
)
PROCESS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "cooling_rate_Cps": (0.1, 50.0),   # areia (~0.1–1) a injeção sob pressão (~20–50)
    "T_solution_C": (480.0, 550.0),
    "T_aging_C": (140.0, 220.0),
    "t_aging_h": (0.5, 24.0),
}
# Features amostradas/normalizadas em escala log10
LOG_FEATURES: Final[frozenset[str]] = frozenset({"cooling_rate_Cps", "t_aging_h"})

INPUT_FEATURES: Final[tuple[str, ...]] = ELEMENTS + PROCESS_FEATURES

# Saídas do modelo
TARGETS: Final[tuple[str, ...]] = ("YS_MPa", "UTS_MPa", "El_pct")
PHASES: Final[tuple[str, ...]] = (
    "f_alpha_Al",          # matriz dendrítica Al-α (com precipitados nanométricos)
    "f_Si_eutectic",       # Si eutético
    "f_beta_AlFeSi",       # β-Al5FeSi (plaquetas/agulhas — nocivo)
    "f_alpha_AlFeMnSi",    # α-Al15(Fe,Mn)3Si2 (escrita chinesa — menos nocivo)
    "f_Al2Cu",             # θ-Al2Cu não dissolvido
)

# Restrições de monotonicidade físicas: (alvo, feature) -> sinal da derivada
MONOTONIC_CONSTRAINTS: Final[dict[tuple[str, str], int]] = {
    ("YS_MPa", "Mg"): +1,                # precipitação de β''-Mg2Si
    ("YS_MPa", "Cu"): +1,                # solução sólida + θ'
    ("El_pct", "Fe"): -1,                # intermetálicos de Fe fragilizam
    ("El_pct", "cooling_rate_Cps"): +1,  # SDAS menor, intermetálicos menores
}

# ---------------------------------------------------------------------------
# Economia e sustentabilidade
# ---------------------------------------------------------------------------
# Preço de referência (USD/kg de elemento contido) das adições.
# Al (primário P1020): LME + prêmio regional. Si: Si metálico 553.
# Mg: Mg 99.9 %. Cu, Zn, Ni: LME. Mn, Ti, Sr, Cr: ligas-mãe (AlMn, AlTi5B1, AlSr10, AlCr).
ELEMENT_PRICE_USD_PER_KG: Final[dict[str, float]] = {
    "Al": 2.65, "Si": 2.20, "Fe": 0.60, "Cu": 9.40, "Mn": 2.40, "Mg": 3.20,
    "Zn": 2.80, "Cr": 9.50, "Ni": 16.50, "Ti": 12.00, "Sr": 30.00,
}

# Fator de emissão cradle-to-gate (kg CO2e / kg de elemento).
# Al primário: média global IAI (~16); use PRIMARY_AL_CO2_LOW_CARBON para
# alumínio de hidroeletricidade. Demais: Nuss & Eckelman (2014), arredondados;
# Mg considera rota Pidgeon.
ELEMENT_CO2_KG_PER_KG: Final[dict[str, float]] = {
    "Al": 16.0, "Si": 6.0, "Fe": 1.5, "Cu": 2.8, "Mn": 1.0, "Mg": 25.0,
    "Zn": 3.1, "Cr": 2.4, "Ni": 6.5, "Ti": 8.1, "Sr": 3.0,
}
PRIMARY_AL_CO2_LOW_CARBON: Final[float] = 4.0
# Refusão de sucata (forno + escória/dross), kg CO2e/kg de metal carregado
REMELT_CO2_KG_PER_KG: Final[float] = 0.5
REMELT_COST_USD_PER_KG: Final[float] = 0.12

# Recuperação metálica das adições no banho (perdas por oxidação/escória)
ELEMENT_RECOVERY: Final[dict[str, float]] = {
    "Al": 0.98, "Si": 0.95, "Fe": 0.95, "Cu": 0.99, "Mn": 0.95, "Mg": 0.85,
    "Zn": 0.95, "Cr": 0.95, "Ni": 0.99, "Ti": 0.85, "Sr": 0.70,
}


@dataclass(frozen=True)
class ScrapGrade:
    """Classe de sucata: composição média (wt%), preço, CO2 e rendimento metálico."""

    name: str
    composition: dict[str, float]
    price_USD_per_kg: float
    co2_kg_per_kg: float = REMELT_CO2_KG_PER_KG
    metal_yield: float = 0.92  # fração recuperada como metal líquido (dross, revestimentos)


# Composições típicas (indicativas) — substitua pela análise OES do seu pátio.
SCRAP_GRADES: Final[dict[str, ScrapGrade]] = {
    "tense_cast": ScrapGrade(  # sucata fundida fragmentada (ISRI "Tense")
        "Tense (fundidos automotivos)",
        {"Si": 8.5, "Fe": 0.95, "Cu": 2.5, "Mn": 0.30, "Mg": 0.30, "Zn": 1.20,
         "Cr": 0.05, "Ni": 0.10, "Ti": 0.05, "Sr": 0.0},
        1.70, metal_yield=0.90,
    ),
    "zorba": ScrapGrade(  # mistura não ferrosa triturada (ISRI "Zorba")
        "Zorba (mista)",
        {"Si": 3.5, "Fe": 1.10, "Cu": 1.50, "Mn": 0.40, "Mg": 0.60, "Zn": 1.50,
         "Cr": 0.08, "Ni": 0.08, "Ti": 0.04, "Sr": 0.0},
        1.45, metal_yield=0.85,
    ),
    "wheels_a356": ScrapGrade(  # rodas de liga (ISRI "Troma")
        "Rodas A356",
        {"Si": 7.0, "Fe": 0.15, "Cu": 0.02, "Mn": 0.02, "Mg": 0.35, "Zn": 0.03,
         "Cr": 0.0, "Ni": 0.0, "Ti": 0.12, "Sr": 0.01},
        2.05, metal_yield=0.95,
    ),
    "ubc": ScrapGrade(  # latas (3104/5182)
        "UBC (latas)",
        {"Si": 0.30, "Fe": 0.50, "Cu": 0.15, "Mn": 0.90, "Mg": 1.20, "Zn": 0.10,
         "Cr": 0.02, "Ni": 0.0, "Ti": 0.02, "Sr": 0.0},
        1.55, metal_yield=0.88,
    ),
    "extrusion_6063": ScrapGrade(  # perfis 6063 limpos (ISRI "Tread")
        "Perfis 6063",
        {"Si": 0.45, "Fe": 0.20, "Cu": 0.05, "Mn": 0.05, "Mg": 0.60, "Zn": 0.05,
         "Cr": 0.05, "Ni": 0.0, "Ti": 0.02, "Sr": 0.0},
        2.15, metal_yield=0.96,
    ),
}

# Alumínio primário (diluente) — lingote P1020
PRIMARY_AL_COMPOSITION: Final[dict[str, float]] = {"Si": 0.06, "Fe": 0.10}

# ---------------------------------------------------------------------------
# Limites metalúrgicos (filtro termodinâmico / risco)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetallurgicalLimits:
    """Limites usados pelo filtro termodinâmico e de risco."""

    min_mn_fe_ratio: float = 0.5            # neutralização do β-Al5FeSi
    mn_fe_ratio_applies_above_fe: float = 0.45
    holding_temperature_C: float = 680.0    # forno de espera
    sludge_factor_base: float = 1.8         # SF crítico a 650 °C (Gobrecht/Jorstad)
    sludge_factor_slope_per_C: float = 0.008  # SF crítico cresce com T de espera
    incipient_melting_margin_C: float = 5.0
    min_dissolution_ratio: float = 0.85     # fração de Cu+Mg dissolvida na solubilização
    max_zn: float = 1.2                     # trincas a quente / corrosão / normas EN AC
    max_cu_corrosion: float = 1.0           # corrosão (aplica-se se especificada)
    max_freezing_range_C: float = 110.0     # porosidade de contração / hot tearing
    max_sr: float = 0.035                   # sobremodificação, porosidade
    max_ti: float = 0.15                    # TiAl3 primário
    max_ni: float = 0.25
    max_si_hypoeutectic: float = 12.0       # validade do modelo (Si primário acima)
    max_undesired_phase_fraction: float = 0.015  # β-AlFeSi/sludge previstos por CALPHAD
    max_beta_fraction: float = 0.02


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
    min_ys_MPa: float = 0.0
    min_uts_MPa: float = 260.0
    min_elongation_pct: float = 4.0
    min_recycled_fraction: float = 0.0   # fração mínima de sucata na carga
    max_risk_score: float = 1.0
    max_uts_cv: float | None = None      # coef. de variação máx. (MC-dropout)
    n_mc_samples: int = 0                # 0 = predição determinística
    scrap_mix: dict[str, float] = field(default_factory=lambda: {"tense_cast": 1.0})
    low_carbon_primary: bool = False     # Al primário de hidroeletricidade
    fixed_values: dict[str, float] = field(default_factory=dict)
