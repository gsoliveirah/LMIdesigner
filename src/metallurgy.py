"""Relações metalúrgicas empíricas compartilhadas.

Todas as funções aceitam um mapeamento ``elemento -> array`` cujos valores
podem ser ``float``, ``numpy.ndarray``, ``pandas.Series`` ou ``torch.Tensor``.
Isso permite usar *as mesmas equações* na geração de dados sintéticos
(NumPy) e nos resíduos físicos diferenciáveis da PINN (PyTorch).

Composições em wt%, temperaturas em °C, tamanho de grão em µm.

Referências principais
----------------------
* Andrews, K.W. (1965) JISI 203, 721 — Ac1, Ac3, Ms.
* Steven, W.; Haynes, A.G. (1956) JISI 183, 349 — Bs.
* Koistinen, D.P.; Marburger, R.E. (1959) Acta Metall. 7, 59.
* Pickering, F.B. (1978) Physical Metallurgy and the Design of Steels.
* Ito, Y.; Bessyo, K. (1968) IIW Doc. IX-576-68 — Pcm.
* Krauss, G. (1999) Mater. Sci. Eng. A 273–275, 40 — resistência da martensita.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar, Union

import numpy as np

try:  # torch é opcional para este módulo
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from src.config import CEMENTITE_C_WT, KM_ALPHA, ROOM_TEMPERATURE_C

ArrayLike = Union[float, np.ndarray, "torch.Tensor", Any]
T = TypeVar("T")
Composition = Mapping[str, ArrayLike]


# ---------------------------------------------------------------------------
# Helpers agnósticos ao backend (NumPy / PyTorch)
# ---------------------------------------------------------------------------
def _is_torch(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _clip(x: ArrayLike, lo: float | None = None, hi: float | None = None) -> ArrayLike:
    if _is_torch(x):
        return torch.clamp(x, min=lo, max=hi)
    return np.clip(x, lo, hi)


def _exp(x: ArrayLike) -> ArrayLike:
    return torch.exp(x) if _is_torch(x) else np.exp(x)


def _log10(x: ArrayLike) -> ArrayLike:
    return torch.log10(x) if _is_torch(x) else np.log10(x)


def _sigmoid(x: ArrayLike) -> ArrayLike:
    if _is_torch(x):
        return torch.sigmoid(x)
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _maximum(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    if _is_torch(a) or _is_torch(b):
        a_t = a if _is_torch(a) else torch.as_tensor(a, dtype=b.dtype, device=b.device)
        b_t = b if _is_torch(b) else torch.as_tensor(b, dtype=a_t.dtype, device=a_t.device)
        return torch.maximum(a_t, b_t)
    return np.maximum(a, b)


def _g(comp: Composition, el: str) -> ArrayLike:
    """Obtém o teor de um elemento (0 se ausente)."""
    value = comp.get(el, 0.0) if hasattr(comp, "get") else 0.0
    return 0.0 if value is None else value


# ---------------------------------------------------------------------------
# Índices de soldabilidade
# ---------------------------------------------------------------------------
def carbon_equivalent_iiw(comp: Composition) -> ArrayLike:
    """Carbono equivalente IIW: C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15."""
    return (
        _g(comp, "C") + _g(comp, "Mn") / 6.0
        + (_g(comp, "Cr") + _g(comp, "Mo") + _g(comp, "V")) / 5.0
        + (_g(comp, "Ni") + _g(comp, "Cu")) / 15.0
    )


def pcm_ito_bessyo(comp: Composition) -> ArrayLike:
    """Parâmetro de trinca Pcm (Ito–Bessyo), adequado para C < 0.18 wt%."""
    return (
        _g(comp, "C") + _g(comp, "Si") / 30.0
        + (_g(comp, "Mn") + _g(comp, "Cu") + _g(comp, "Cr")) / 20.0
        + _g(comp, "Ni") / 60.0 + _g(comp, "Mo") / 15.0 + _g(comp, "V") / 10.0
        + 5.0 * _g(comp, "B")
    )


def schaeffler_equivalents(comp: Composition) -> tuple[ArrayLike, ArrayLike]:
    """Retorna (Cr_eq, Ni_eq) de Schaeffler."""
    cr_eq = _g(comp, "Cr") + _g(comp, "Mo") + 1.5 * _g(comp, "Si") + 0.5 * _g(comp, "Nb")
    ni_eq = _g(comp, "Ni") + 30.0 * _g(comp, "C") + 0.5 * _g(comp, "Mn")
    return cr_eq, ni_eq


# ---------------------------------------------------------------------------
# Temperaturas críticas
# ---------------------------------------------------------------------------
def ac1_andrews(comp: Composition) -> ArrayLike:
    """Ac1 (°C) — Andrews (1965)."""
    return (
        723.0 - 10.7 * _g(comp, "Mn") - 16.9 * _g(comp, "Ni")
        + 29.1 * _g(comp, "Si") + 16.9 * _g(comp, "Cr")
    )


def ac3_andrews(comp: Composition) -> ArrayLike:
    """Ac3 (°C) — Andrews (1965), forma estendida."""
    c = _clip(_g(comp, "C"), 0.0, None)
    return (
        910.0 - 203.0 * c**0.5 - 15.2 * _g(comp, "Ni") + 44.7 * _g(comp, "Si")
        + 104.0 * _g(comp, "V") + 31.5 * _g(comp, "Mo") - 30.0 * _g(comp, "Mn")
        - 11.0 * _g(comp, "Cr") - 20.0 * _g(comp, "Cu") + 400.0 * _g(comp, "Al")
        + 400.0 * _g(comp, "Ti")
    )


def ms_andrews(comp: Composition, carbon: ArrayLike | None = None) -> ArrayLike:
    """Temperatura Ms (°C) — Andrews linear.

    ``carbon`` permite usar o teor de C particionado na austenita.
    """
    c = _g(comp, "C") if carbon is None else carbon
    return (
        539.0 - 423.0 * c - 30.4 * _g(comp, "Mn") - 17.7 * _g(comp, "Ni")
        - 12.1 * _g(comp, "Cr") - 7.5 * _g(comp, "Mo") - 7.5 * _g(comp, "Si")
    )


def bs_steven_haynes(comp: Composition, carbon: ArrayLike | None = None) -> ArrayLike:
    """Temperatura Bs (°C) — Steven & Haynes."""
    c = _g(comp, "C") if carbon is None else carbon
    return (
        830.0 - 270.0 * c - 90.0 * _g(comp, "Mn") - 37.0 * _g(comp, "Ni")
        - 70.0 * _g(comp, "Cr") - 83.0 * _g(comp, "Mo")
    )


def log10_critical_cooling_rate(comp: Composition, carbon: ArrayLike | None = None) -> ArrayLike:
    """log10 da taxa crítica de resfriamento para martensita (°C/s).

    Modelo de temperabilidade parametrizado (forma de Kasatkin simplificada),
    incluindo o efeito sinérgico do B livre. Valores indicativos.
    """
    c = _g(comp, "C") if carbon is None else carbon
    b_eff = _clip(_g(comp, "B") * 2000.0, 0.0, 1.0)  # saturação ~5 ppm
    return (
        3.8 - 2.6 * c - 0.75 * _g(comp, "Mn") - 0.35 * _g(comp, "Cr")
        - 0.50 * _g(comp, "Mo") - 0.25 * _g(comp, "Ni") - 0.10 * _g(comp, "Si")
        - 0.40 * b_eff
    )


def koistinen_marburger(ms: ArrayLike, quench_temp_C: float = ROOM_TEMPERATURE_C) -> ArrayLike:
    """Fração de austenita transformada em martensita até ``quench_temp_C``."""
    undercooling = _clip(ms - quench_temp_C, 0.0, None)
    return 1.0 - _exp(-KM_ALPHA * undercooling)


# ---------------------------------------------------------------------------
# Resistência dos constituintes
# ---------------------------------------------------------------------------
def ferrite_yield_strength(comp: Composition, grain_size_um: ArrayLike) -> ArrayLike:
    """Limite de escoamento da ferrita (MPa) — Pickering + precipitação.

    σy = 53.9 + 32.3Mn + 83.2Si + 354√N_livre + 17.4 d^-1/2 (d em mm)
         + Δσ_ppt(Nb, Ti, V) + contribuições menores (Cr, Ni, Cu, Mo).
    """
    d_mm = _clip(grain_size_um, 0.5, None) / 1000.0
    n_free = _clip(_g(comp, "N") - 0.29 * _g(comp, "Ti"), 0.0, None)
    ppt = _clip(
        1500.0 * _g(comp, "Nb") + 900.0 * _g(comp, "Ti") + 700.0 * _g(comp, "V"),
        0.0, 180.0,
    )
    return (
        53.9 + 32.3 * _g(comp, "Mn") + 83.2 * _g(comp, "Si") + 354.0 * (n_free + 1e-8) ** 0.5
        + 17.4 * d_mm**-0.5 + ppt + 30.0 * _g(comp, "Cr") + 33.0 * _g(comp, "Ni")
        + 39.0 * _g(comp, "Cu") + 11.0 * _g(comp, "Mo")
    )


def martensite_uts(carbon_in_martensite: ArrayLike, comp: Composition) -> ArrayLike:
    """UTS da martensita virgem (MPa) — dependência ~linear com C (Krauss)."""
    c = _clip(carbon_in_martensite, 0.0, 0.8)
    return 800.0 + 2900.0 * c + 40.0 * _g(comp, "Mn") + 30.0 * _g(comp, "Cr")


def constituent_strengths(
    comp: Composition, grain_size_um: ArrayLike, carbon_hard: ArrayLike
) -> dict[str, tuple[ArrayLike, ArrayLike]]:
    """Retorna {fase: (YS, UTS)} em MPa para cada constituinte."""
    ys_f = ferrite_yield_strength(comp, grain_size_um)
    uts_f = 1.25 * ys_f + 80.0
    uts_m = martensite_uts(carbon_hard, comp)
    ys_m = 0.55 * uts_m  # escoamento contínuo em aços multifásicos
    uts_b = 0.55 * uts_m + 100.0
    ys_b = 0.80 * uts_b
    uts_ra = 0.90 * uts_m  # efeito TRIP: RA transforma em martensita fresca
    ys_ra = 350.0 + 0.0 * uts_m
    return {
        "f_ferrite": (ys_f, uts_f),
        "f_bainite": (ys_b, uts_b),
        "f_martensite": (ys_m, uts_m),
        "f_retained_austenite": (ys_ra, uts_ra),
    }


def hard_phase_carbon(comp: Composition, f_hard: ArrayLike) -> ArrayLike:
    """Balanço de massa de C: o carbono particiona para as fases duras."""
    return _clip(_g(comp, "C") / _clip(f_hard, 0.05, 1.0), 0.0, 1.2)


def rule_of_mixtures(
    comp: Composition, phases: Mapping[str, ArrayLike], grain_size_um: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """(YS, UTS) previstos por regra das misturas a partir das frações de fase."""
    f_hard = (
        phases["f_bainite"] + phases["f_martensite"] + phases["f_retained_austenite"]
    )
    c_hard = hard_phase_carbon(comp, f_hard)
    strengths = constituent_strengths(comp, grain_size_um, c_hard)
    ys = 0.0
    uts = 0.0
    for name, (ys_i, uts_i) in strengths.items():
        ys = ys + phases[name] * ys_i
        uts = uts + phases[name] * uts_i
    return ys, uts


def elongation_model(uts: ArrayLike, f_retained_austenite: ArrayLike) -> ArrayLike:
    """Alongamento total (%) — compromisso resistência-ductilidade + TRIP."""
    raw = 48.0 - 0.035 * uts + 50.0 * f_retained_austenite
    return _maximum(raw, 3.0 + 0.0 * raw)


# ---------------------------------------------------------------------------
# Modelo de microestrutura (recozimento intercrítico + resfriamento)
# ---------------------------------------------------------------------------
def simulate_microstructure(
    comp: Composition,
    T_anneal_C: ArrayLike,
    cooling_rate_Cps: ArrayLike,
    austenite_fraction: ArrayLike | None = None,
    austenite_carbon: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Simula a microestrutura final (NumPy).

    Etapas:
      1. Fração de austenita intercrítica (CALPHAD se fornecida, senão
         regra da alavanca linear entre Ac1 e Ac3).
      2. Particionamento de C para a austenita (balanço de massa).
      3. Decomposição no resfriamento: ferrita/perlita, bainita ou
         martensita conforme a taxa crítica (temperabilidade).
      4. Estabilização de austenita por Si+Al no estágio bainítico (TRIP).
      5. Koistinen–Marburger para a martensita; o resto é austenita retida.
    """
    T = np.asarray(T_anneal_C, dtype=float)
    cr = np.clip(np.asarray(cooling_rate_Cps, dtype=float), 1e-3, None)
    ac1 = np.asarray(ac1_andrews(comp), dtype=float)
    ac3 = np.asarray(ac3_andrews(comp), dtype=float)

    if austenite_fraction is None:
        f_gamma = np.clip((T - ac1) / np.clip(ac3 - ac1, 1.0, None), 0.0, 1.0)
    else:
        f_gamma = np.clip(np.asarray(austenite_fraction, dtype=float), 0.0, 1.0)

    c_bulk = np.asarray(_g(comp, "C"), dtype=float)
    if austenite_carbon is None:
        c_gamma = np.clip(c_bulk / np.clip(f_gamma, 0.05, 1.0), 0.0, 1.2)
    else:
        c_gamma = np.clip(np.asarray(austenite_carbon, dtype=float), 0.0, 1.2)

    log_cr = np.log10(cr)
    log_crit = np.asarray(log10_critical_cooling_rate(comp, carbon=c_gamma), dtype=float)
    p_mart = _sigmoid((log_cr - log_crit) / 0.25)
    p_bain = np.clip(_sigmoid((log_cr - (log_crit - 1.0)) / 0.25) - p_mart, 0.0, 1.0)
    p_fp = np.clip(1.0 - p_mart - p_bain, 0.0, 1.0)

    si_al = np.asarray(_g(comp, "Si"), dtype=float) + np.asarray(_g(comp, "Al"), dtype=float)
    r_stab = 0.25 * np.clip(si_al / 1.5, 0.0, 1.0)

    ms = np.asarray(ms_andrews(comp, carbon=c_gamma), dtype=float)
    km = np.asarray(koistinen_marburger(ms), dtype=float)

    f_m_pool = f_gamma * p_mart
    f_mart = f_m_pool * km
    f_ra = (f_m_pool - f_mart) + f_gamma * p_bain * r_stab
    f_bain = f_gamma * p_bain * (1.0 - r_stab)
    f_ferr = (1.0 - f_gamma) + f_gamma * p_fp

    total = f_ferr + f_bain + f_mart + f_ra
    return {
        "f_ferrite": f_ferr / total,
        "f_bainite": f_bain / total,
        "f_martensite": f_mart / total,
        "f_retained_austenite": f_ra / total,
        "f_austenite_IA": f_gamma,
        "C_austenite_wt": c_gamma,
        "Ac1_C": ac1,
        "Ac3_C": ac3,
        "Ms_austenite_C": ms,
        "f_cementite_eq": np.clip((1.0 - f_gamma) * c_bulk / CEMENTITE_C_WT, 0.0, 1.0),
    }
