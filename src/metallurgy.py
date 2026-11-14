"""Relações metalúrgicas compartilhadas para ligas de fundição Al-Si secundárias.

Todas as funções aceitam um mapeamento ``elemento -> array`` cujos valores
podem ser ``float``, ``numpy.ndarray``, ``pandas.Series`` ou ``torch.Tensor``.
Assim, *as mesmas equações* servem à geração de dados sintéticos (NumPy) e
aos resíduos físicos diferenciáveis da PINN (PyTorch).

Composições em wt% (Al = balanço), temperaturas em °C, tempos em h,
taxa de resfriamento em °C/s, SDAS em µm, frações de fase em fração mássica.

Cadeia de modelos
-----------------
composição + taxa de resfriamento
    -> SDAS (lei de potência) e partição do Fe entre β-Al5FeSi e α-Al15(Fe,Mn)3Si2
composição + T de solubilização
    -> Cu e Mg dissolvidos (solvus de Arrhenius), Al2Cu residual, fusão incipiente
T e t de envelhecimento
    -> parâmetro cinético P de Shercliff–Ashby -> endurecimento por precipitação
fases + solutos + SDAS
    -> YS (superposição de mecanismos), alongamento (dano por intermetálicos),
       UTS (encruamento até a estricção)

Referências principais
----------------------
* Taylor, J.A. (2004) Proc. 3rd Int. Conf. Light Metals Tech. — Fe crítico.
* Shabestari, S.G. (2004) MSE A 383, 289 — sludge factor (Fe + 2Mn + 3Cr).
* Djurdjevic, M. et al. (2001) — Si equivalente e temperatura liquidus.
* Shercliff, H.R.; Ashby, M.F. (1990) Acta Metall. Mater. 38, 1789.
* Cáceres, C.H.; Taylor, J.A. (2006) MMTA 37, 3393 — ductilidade de Al-Si-Mg.
* Wang, Q.G. (2003) MMTA 34, 2887 — SDAS, Si eutético e propriedades de A356/A357.
* Mondolfo, L.F. (1976) Aluminum Alloys: Structure and Properties.

Os parâmetros são *semiquantitativos*: suficientes para dados sintéticos
fisicamente coerentes e para regularizar a PINN, não para projeto final sem
calibração experimental.
"""

from __future__ import annotations

from typing import Any, Mapping, Union

import numpy as np

try:  # torch é opcional para este módulo
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from src.config import (
    AGING_ACTIVATION_J,
    AGING_PEAK_TIME_REF_H,
    AGING_REF_C,
    AL2CU_CU_WT,
    AL_MELTING_C,
    AL_SI_CU_EUTECTIC_C,
    AL_SI_CU_MG_EUTECTIC_C,
    AL_SI_EUTECTIC_C,
    ALPHA_FEMN_WT,
    ALPHA_SI_WT,
    BETA_FE_WT,
    BETA_SI_WT,
    CU_DISSOLUTION_H,
    CU_MAX_SOLUBILITY_WT,
    CU_SOLVUS_REF_C,
    KELVIN_OFFSET,
    MG_DISSOLUTION_H,
    MG_MAX_SOLUBILITY_WT,
    MG_SOLVUS_REF_C,
    R_GAS,
    SDAS_A_UM,
    SDAS_N,
    SI_IN_MATRIX_WT,
)

ArrayLike = Union[float, np.ndarray, "torch.Tensor", Any]
Composition = Mapping[str, ArrayLike]

# Solubilidade residual de Fe no Al-α (wt%) — praticamente insolúvel
FE_SOLUBLE_WT: float = 0.03


# ---------------------------------------------------------------------------
# Helpers agnósticos ao backend (NumPy / PyTorch)
# ---------------------------------------------------------------------------
def _is_torch(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _any_torch(*xs: Any) -> Any:
    for x in xs:
        if _is_torch(x):
            return x
    return None


def _clip(x: ArrayLike, lo: float | None = None, hi: float | None = None) -> ArrayLike:
    if _is_torch(x):
        return torch.clamp(x, min=lo, max=hi)
    return np.clip(x, lo, hi)


def _exp(x: ArrayLike) -> ArrayLike:
    return torch.exp(x) if _is_torch(x) else np.exp(x)


def _log10(x: ArrayLike) -> ArrayLike:
    return torch.log10(x) if _is_torch(x) else np.log10(x)


def _sqrt(x: ArrayLike) -> ArrayLike:
    return torch.sqrt(x) if _is_torch(x) else np.sqrt(x)


def _sigmoid(x: ArrayLike) -> ArrayLike:
    if _is_torch(x):
        return torch.sigmoid(x)
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _to_common(a: ArrayLike, b: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    ref = _any_torch(a, b)
    if ref is None:
        return a, b
    a_t = a if _is_torch(a) else torch.as_tensor(a, dtype=ref.dtype, device=ref.device)
    b_t = b if _is_torch(b) else torch.as_tensor(b, dtype=ref.dtype, device=ref.device)
    return a_t, b_t


def _minimum(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    a, b = _to_common(a, b)
    return torch.minimum(a, b) if _is_torch(a) else np.minimum(a, b)


def _maximum(a: ArrayLike, b: ArrayLike) -> ArrayLike:
    a, b = _to_common(a, b)
    return torch.maximum(a, b) if _is_torch(a) else np.maximum(a, b)


def _g(comp: Composition, el: str) -> ArrayLike:
    """Teor de um elemento (0 se ausente)."""
    value = comp.get(el, 0.0) if hasattr(comp, "get") else 0.0
    return 0.0 if value is None else value


def _ramp(x: ArrayLike, scale: float) -> ArrayLike:
    """Rampa saturada 0→1 em [0, scale]."""
    return _clip(x / scale, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Equivalentes composicionais e temperaturas características
# ---------------------------------------------------------------------------
# Inclinações liquidus dos binários Al-X no canto rico em Al (°C/wt%).
# Si equivalente de X = (m_X / m_Si) · X  (Cr e Ti: peritéticos, efeito ~0).
LIQUIDUS_SLOPES: dict[str, float] = {
    "Si": -6.6, "Cu": -3.4, "Mg": -5.6, "Fe": -2.8, "Mn": -1.0, "Zn": -1.3,
    "Ni": -3.3, "Cr": 0.0, "Ti": 0.0, "Sr": 0.0,
}


def silicon_equivalent(comp: Composition) -> ArrayLike:
    """Si equivalente (Si_EQ, wt%) — análogo, para Al-Si, do carbono equivalente dos aços.

    Converte cada soluto no teor de Si que produziria o mesmo abaixamento da
    liquidus (abordagem de Djurdjevic et al.). Governa a temperatura liquidus,
    a fluidez e a proximidade do eutético.
    """
    si_eq = _g(comp, "Si")
    m_si = LIQUIDUS_SLOPES["Si"]
    for el, m in LIQUIDUS_SLOPES.items():
        if el != "Si" and m != 0.0:
            si_eq = si_eq + (m / m_si) * _g(comp, el)
    return si_eq


def liquidus_temperature(comp: Composition) -> ArrayLike:
    """Temperatura liquidus (°C) — polinômio Al-Si de Djurdjevic em Si_EQ."""
    si_eq = _clip(silicon_equivalent(comp), 0.0, None)
    return AL_MELTING_C - 6.11780 * si_eq - 0.057709 * si_eq**2


def incipient_melting_temperature(comp: Composition) -> ArrayLike:
    """Temperatura do último eutético a solidificar (°C) — limite da solubilização.

    * Al-Si(-Mg): eutético binário a 577 °C.
    * Cu ≳ 0.5 %: eutético Al-Si-Al2Cu a ~525 °C (segregação de solidificação).
    * Cu + Mg: eutético quaternário com fase Q a ~507 °C.
    * Zn abaixa marginalmente.
    """
    s_cu = _ramp(_g(comp, "Cu"), 0.6)
    s_mg = _ramp(_g(comp, "Mg"), 0.25)
    return (
        AL_SI_EUTECTIC_C
        - (AL_SI_EUTECTIC_C - AL_SI_CU_EUTECTIC_C) * s_cu
        - (AL_SI_CU_EUTECTIC_C - AL_SI_CU_MG_EUTECTIC_C) * s_cu * s_mg
        - 2.0 * _g(comp, "Zn")
    )


def freezing_range(comp: Composition) -> ArrayLike:
    """Intervalo de solidificação (°C): maior -> mais porosidade e trinca a quente."""
    return liquidus_temperature(comp) - incipient_melting_temperature(comp)


# ---------------------------------------------------------------------------
# Química do Fe (impureza-chave da reciclagem)
# ---------------------------------------------------------------------------
def sludge_factor(comp: Composition) -> ArrayLike:
    """Sludge factor SF = Fe + 2Mn + 3Cr (wt%)."""
    return _g(comp, "Fe") + 2.0 * _g(comp, "Mn") + 3.0 * _g(comp, "Cr")


def critical_sludge_factor(holding_temperature_C: ArrayLike, base: float = 1.8,
                           slope: float = 0.008) -> ArrayLike:
    """SF acima do qual intermetálicos primários (sludge) precipitam no forno."""
    return base + slope * (holding_temperature_C - 650.0)


def critical_iron(comp: Composition) -> ArrayLike:
    """Fe crítico (wt%) de Taylor: Fe_crit ≈ 0.075·(Si − 4.05).

    Acima dele o β-Al5FeSi cristaliza *antes* do eutético Al-Si, formando
    plaquetas grosseiras que bloqueiam a alimentação (porosidade) e trincam.
    """
    return _clip(0.075 * (_g(comp, "Si") - 4.05), 0.05, None)


def mn_fe_ratio(comp: Composition) -> ArrayLike:
    """Razão (Mn + 2Cr)/Fe: ≥ 0.5 promove α-Al15(Fe,Mn)3Si2 em vez de β."""
    return (_g(comp, "Mn") + 2.0 * _g(comp, "Cr")) / _clip(_g(comp, "Fe"), 1e-3, None)


def alpha_fe_fraction(comp: Composition, cooling_rate_Cps: ArrayLike) -> ArrayLike:
    """Fração do Fe intermetálico que forma α (escrita chinesa) em vez de β.

    O limiar da razão Mn/Fe diminui com a taxa de resfriamento (resfriamento
    rápido suprime o β).
    """
    log_cr = _clip(_log10(_clip(cooling_rate_Cps, 1e-3, None)), -1.0, 1.7)
    threshold = 0.45 - 0.12 * log_cr
    return _sigmoid((mn_fe_ratio(comp) - threshold) / 0.08)


# ---------------------------------------------------------------------------
# Solidificação e solubilização
# ---------------------------------------------------------------------------
def sdas_um(cooling_rate_Cps: ArrayLike) -> ArrayLike:
    """Espaçamento dendrítico secundário (µm): SDAS = A·CR^(−n)."""
    cr = _clip(cooling_rate_Cps, 1e-3, None)
    return SDAS_A_UM * _exp(-SDAS_N * _log10(cr) * np.log(10.0))


def _arrhenius_solubility(T_C: ArrayLike, c_max: float, T_ref_C: float, dH: float) -> ArrayLike:
    T_K = _clip(T_C, 300.0, T_ref_C) + KELVIN_OFFSET
    return c_max * _exp(-dH / R_GAS * (1.0 / T_K - 1.0 / (T_ref_C + KELVIN_OFFSET)))


def cu_solubility(T_C: ArrayLike) -> ArrayLike:
    """Solubilidade de Cu no Al-α (wt%) — solvus de Arrhenius (máx. 5.65 % a 548 °C)."""
    return _arrhenius_solubility(T_C, CU_MAX_SOLUBILITY_WT, CU_SOLVUS_REF_C, CU_DISSOLUTION_H)


def mg_solubility(T_C: ArrayLike) -> ArrayLike:
    """Solubilidade de Mg no Al-α saturado em Si (wt%)."""
    return _arrhenius_solubility(T_C, MG_MAX_SOLUBILITY_WT, MG_SOLVUS_REF_C, MG_DISSOLUTION_H)


def dissolved_solutes(comp: Composition, T_solution_C: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """(Cu, Mg) dissolvidos (wt%) após solubilização (tempo suficiente, equilíbrio local)."""
    cu = _minimum(_g(comp, "Cu"), cu_solubility(T_solution_C))
    mg = _minimum(_g(comp, "Mg"), mg_solubility(T_solution_C))
    return cu, mg


def incipient_melting_damage(comp: Composition, T_solution_C: ArrayLike) -> ArrayLike:
    """Severidade (0–1) de fusão incipiente ("queima") na solubilização."""
    return _sigmoid((T_solution_C - incipient_melting_temperature(comp)) / 3.0)


# ---------------------------------------------------------------------------
# Envelhecimento artificial
# ---------------------------------------------------------------------------
def aging_parameter(T_aging_C: ArrayLike, t_aging_h: ArrayLike, comp: Composition | None = None) -> ArrayLike:
    """Parâmetro cinético P = t_eq / t_pico de Shercliff–Ashby.

    t_eq = t·exp[−Q/R·(1/T − 1/T_ref)]; o Cu retarda a cinética (t_pico maior).
    P < 1: subenvelhecido, P ≈ 1: pico (T6), P ≫ 1: superenvelhecido (T7).
    """
    T_K = T_aging_C + KELVIN_OFFSET
    T_ref = AGING_REF_C + KELVIN_OFFSET
    t_eq = _clip(t_aging_h, 1e-3, None) * _exp(-AGING_ACTIVATION_J / R_GAS * (1.0 / T_K - 1.0 / T_ref))
    t_peak = AGING_PEAK_TIME_REF_H
    if comp is not None:
        t_peak = t_peak * (1.0 + 0.25 * _g(comp, "Cu"))
    return t_eq / t_peak


def aging_response(P: ArrayLike) -> ArrayLike:
    """Fração da resistência de pico: S(P) = [2√P / (1 + P)]^½ (máximo 1 em P = 1).

    O expoente ½ suaviza a queda no superenvelhecimento (coarsening ∝ P^(−1/4)),
    coerente com a perda de ~20–30 % de YS observada em tratamentos T7.
    """
    P = _clip(P, 1e-6, None)
    return _sqrt(2.0 * _sqrt(P) / (1.0 + P))


# ---------------------------------------------------------------------------
# Microestrutura
# ---------------------------------------------------------------------------
def phase_fractions(
    comp: Composition,
    cooling_rate_Cps: ArrayLike,
    T_solution_C: ArrayLike,
    cu_dissolved: ArrayLike | None = None,
) -> dict[str, ArrayLike]:
    """Frações mássicas das fases após T6 (balanço de massa estequiométrico).

    ``cu_dissolved`` permite injetar o Cu dissolvido calculado por CALPHAD.
    """
    fe = _g(comp, "Fe")
    mn_cr = _g(comp, "Mn") + _g(comp, "Cr")
    fe_ip = _clip(fe - FE_SOLUBLE_WT, 0.0, None)        # Fe em intermetálicos
    x_alpha = alpha_fe_fraction(comp, cooling_rate_Cps)
    fe_alpha = fe_ip * x_alpha
    fe_beta = fe_ip - fe_alpha
    mn_alpha = _minimum(mn_cr, 0.5 * fe_alpha)          # razão Fe:Mn ~ 2:1 no α
    f_beta = fe_beta / (100.0 * BETA_FE_WT)
    f_alpha = (fe_alpha + mn_alpha) / (100.0 * ALPHA_FEMN_WT)

    if cu_dissolved is None:
        cu_dissolved, _ = dissolved_solutes(comp, T_solution_C)
    cu_residual = _clip(_g(comp, "Cu") - cu_dissolved, 0.0, None)
    f_al2cu = cu_residual / (100.0 * AL2CU_CU_WT)

    si_in_fe_phases = 100.0 * (BETA_SI_WT * f_beta + ALPHA_SI_WT * f_alpha)
    si_in_mg2si = 0.578 * _g(comp, "Mg")
    si_free = _clip(_g(comp, "Si") - SI_IN_MATRIX_WT - si_in_fe_phases - si_in_mg2si, 0.0, None)
    f_si = si_free / 100.0

    f_matrix = 1.0 - f_si - f_beta - f_alpha - f_al2cu
    return {
        "f_alpha_Al": f_matrix,
        "f_Si_eutectic": f_si,
        "f_beta_AlFeSi": f_beta,
        "f_alpha_AlFeMnSi": f_alpha,
        "f_Al2Cu": f_al2cu,
    }


# ---------------------------------------------------------------------------
# Propriedades mecânicas
# ---------------------------------------------------------------------------
def yield_strength(
    comp: Composition,
    phases: Mapping[str, ArrayLike],
    cooling_rate_Cps: ArrayLike,
    T_solution_C: ArrayLike,
    T_aging_C: ArrayLike,
    t_aging_h: ArrayLike,
    mg_dissolved: ArrayLike | None = None,
) -> ArrayLike:
    """Limite de escoamento (MPa) por superposição de mecanismos.

    σy = σ0 + σ_ss + σ_ppt + σ_SDAS + σ_eut, com
      σ_ppt  = S(P)·√[(k_Mg√Mg_d)² + (k_Cu√Cu_d)²]  (β''-Mg2Si e θ'/Q')
      σ_ss   = solutos remanescentes em solução sólida (decrescem com S)
      σ_SDAS = k / √SDAS  (refino da estrutura dendrítica)
      σ_eut  = transferência de carga para o Si eutético
    Cu dissolvido = Cu − 54.1 %·f_Al2Cu (balanço de massa sobre a fase prevista).
    A fusão incipiente reduz a resistência.
    """
    cu_d = _clip(_g(comp, "Cu") - 100.0 * AL2CU_CU_WT * phases["f_Al2Cu"], 0.0, None)
    if mg_dissolved is None:
        _, mg_dissolved = dissolved_solutes(comp, T_solution_C)
    mg_d = _clip(mg_dissolved, 0.0, None)
    S = aging_response(aging_parameter(T_aging_C, t_aging_h, comp))

    s_mg = 260.0 * _sqrt(mg_d + 1e-8)
    s_cu = 70.0 * _sqrt(cu_d + 1e-8)
    sigma_ppt = S * _sqrt(s_mg**2 + s_cu**2)
    sigma_ss = (1.0 - 0.6 * S) * (14.0 * cu_d + 18.0 * mg_d) + 3.0 * _g(comp, "Zn") + 8.0 * _g(comp, "Ni")
    sigma_sdas = 120.0 / _sqrt(sdas_um(cooling_rate_Cps))
    sigma_eut = 80.0 * phases["f_Si_eutectic"] + 150.0 * phases["f_alpha_AlFeMnSi"]
    ys = 20.0 + sigma_ss + sigma_ppt + sigma_sdas + sigma_eut
    return ys * (1.0 - 0.15 * incipient_melting_damage(comp, T_solution_C))


def elongation(
    comp: Composition,
    phases: Mapping[str, ArrayLike],
    cooling_rate_Cps: ArrayLike,
    T_solution_C: ArrayLike,
    ys_MPa: ArrayLike,
) -> ArrayLike:
    """Alongamento total (%) — dano por partículas frágeis + compromisso com a resistência.

    * base ∝ exp(−SDAS/50): estrutura fina retarda a fratura das partículas;
    * β-Al5FeSi: penalidade forte, maior para plaquetas longas (SDAS grosseiro);
    * Si eutético: atenuada por modificação com Sr ou solidificação rápida;
    * α-Fe e Al2Cu residual: penalidades moderadas;
    * fusão incipiente: perda severa.
    """
    sdas = sdas_um(cooling_rate_Cps)
    base = 30.0 * _exp(-sdas / 50.0) + 1.0
    modification = _maximum(_ramp(_g(comp, "Sr"), 0.010),
                            _ramp(_log10(_clip(cooling_rate_Cps, 1e-3, None)), 1.7))
    k_beta = 40.0 * _sqrt(sdas / 30.0)
    damage = (
        k_beta * phases["f_beta_AlFeSi"]
        + 10.0 * phases["f_alpha_AlFeMnSi"]
        + 15.0 * phases["f_Al2Cu"]
        + 8.0 * (1.0 - 0.6 * modification) * phases["f_Si_eutectic"]
    )
    el = base * _exp(-damage) * _exp(-ys_MPa / 350.0)
    el = el * (1.0 - 0.7 * incipient_melting_damage(comp, T_solution_C))
    return _clip(el, 0.3, None)


def ultimate_tensile_strength(ys_MPa: ArrayLike, el_pct: ArrayLike) -> ArrayLike:
    """UTS (MPa) = YS + H·(1 − e^(−El/4)), H = máx(170 − 0.3·YS, 30).

    Ligas fundidas fraturam antes da estricção: quanto menor a ductilidade,
    menor o encruamento acumulado até a fratura (UTS → YS quando El → 0).
    """
    H = _clip(170.0 - 0.3 * ys_MPa, 30.0, None)
    return ys_MPa + H * (1.0 - _exp(-el_pct / 4.0))


def mechanical_properties(
    comp: Composition,
    phases: Mapping[str, ArrayLike],
    cooling_rate_Cps: ArrayLike,
    T_solution_C: ArrayLike,
    T_aging_C: ArrayLike,
    t_aging_h: ArrayLike,
    mg_dissolved: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """(YS, UTS, El) a partir de composição, fases e processo."""
    ys = yield_strength(comp, phases, cooling_rate_Cps, T_solution_C, T_aging_C, t_aging_h,
                        mg_dissolved=mg_dissolved)
    el = elongation(comp, phases, cooling_rate_Cps, T_solution_C, ys)
    uts = ultimate_tensile_strength(ys, el)
    return ys, uts, el


def mass_balance_residuals(comp: Composition, phases: Mapping[str, ArrayLike]) -> dict[str, ArrayLike]:
    """Violações (≥ 0, wt%) do balanço de massa das fases intermetálicas.

    * Fe: praticamente insolúvel → todo o Fe acima de 0.03 % deve estar em β ou α,
      e (Fe+Mn) em α não pode exceder Fe + Mn + Cr disponíveis.
    * Cu: Cu no Al2Cu ≤ Cu total.
    * Si: Si em fases ≤ Si total.
    """
    fe_ip = _clip(_g(comp, "Fe") - FE_SOLUBLE_WT, 0.0, None)
    fe_mn_in_phases = 100.0 * (BETA_FE_WT * phases["f_beta_AlFeSi"] + ALPHA_FEMN_WT * phases["f_alpha_AlFeMnSi"])
    fe_upper = fe_ip + _g(comp, "Mn") + _g(comp, "Cr")
    fe_viol = _clip(fe_ip - fe_mn_in_phases, 0.0, None) + _clip(fe_mn_in_phases - fe_upper, 0.0, None)
    cu_viol = _clip(100.0 * AL2CU_CU_WT * phases["f_Al2Cu"] - _g(comp, "Cu"), 0.0, None)
    si_in_phases = 100.0 * (phases["f_Si_eutectic"] + BETA_SI_WT * phases["f_beta_AlFeSi"]
                            + ALPHA_SI_WT * phases["f_alpha_AlFeMnSi"])
    si_viol = _clip(si_in_phases - _g(comp, "Si"), 0.0, None)
    return {"Fe": fe_viol, "Cu": cu_viol, "Si": si_viol}


# ---------------------------------------------------------------------------
# Simulação completa (NumPy)
# ---------------------------------------------------------------------------
def simulate_casting(
    comp: Composition,
    cooling_rate_Cps: ArrayLike,
    T_solution_C: ArrayLike,
    T_aging_C: ArrayLike,
    t_aging_h: ArrayLike,
    cu_dissolved: ArrayLike | None = None,
    mg_dissolved: ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    """Simula microestrutura e propriedades (sem ruído) de uma peça fundida + T6.

    ``cu_dissolved``/``mg_dissolved`` permitem usar solutos do equilíbrio CALPHAD.
    """
    cr = np.asarray(cooling_rate_Cps, dtype=float)
    Ts = np.asarray(T_solution_C, dtype=float)
    Ta = np.asarray(T_aging_C, dtype=float)
    ta = np.asarray(t_aging_h, dtype=float)
    phases = {k: np.asarray(v, float) for k, v in phase_fractions(comp, cr, Ts, cu_dissolved).items()}
    if mg_dissolved is None:
        _, mg_dissolved = dissolved_solutes(comp, Ts)
    ys, uts, el = mechanical_properties(comp, phases, cr, Ts, Ta, ta, mg_dissolved=mg_dissolved)
    return {
        **phases,
        "YS_MPa": np.asarray(ys, float),
        "UTS_MPa": np.asarray(uts, float),
        "El_pct": np.asarray(el, float),
        "SDAS_um": np.asarray(sdas_um(cr), float),
        "aging_P": np.asarray(aging_parameter(Ta, ta, comp), float),
        "Mg_dissolved_wt": np.asarray(mg_dissolved, float) * np.ones_like(cr),
        "melting_damage": np.asarray(incipient_melting_damage(comp, Ts), float),
    }
