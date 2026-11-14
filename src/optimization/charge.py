"""Modelo de carga do forno: sucata + alumínio primário (diluente) + adições.

O refino de alumínio reciclado tem uma assimetria fundamental: elementos em
*falta* são corrigidos por adição (Si metálico, Mg, ligas-mãe AlMn/AlSr/AlTi),
mas elementos em *excesso* (Fe, Cu, Zn, Ni, Cr) praticamente não podem ser
removidos do banho — apenas diluídos com alumínio primário, caro e
carbono-intensivo. Assim, a composição-alvo determina a fração máxima de
sucata na carga, e daí o custo e a pegada de CO2.

Dois níveis:

* :class:`ChargeModel` — mistura de sucata em proporções fixas (``scrap_mix``)
  + primário + adições puras. Fração de sucata máxima em forma fechada,
  vetorizada: usada *dentro* do laço do NSGA-III.
* :func:`optimize_charge_lp` — programação linear (HiGHS) sobre várias classes
  de sucata simultaneamente, minimizando custo ou CO2. Usada no conjunto de
  Pareto final e no dashboard.

Balanço de massa (1 kg de metal líquido)::

    Σ_g x_g·c_g,e + x_p·c_p,e + 100·a_e = c_t,e   ∀ elemento e
    Σ_g x_g + x_p + Σ_e a_e = 1,   x, a ≥ 0

(as adições são tratadas como elemento contido; o veículo Al das ligas-mãe é
contabilizado como alumínio primário, aproximação usual em cálculo de carga).

Elementos não adicionáveis (:data:`~src.config.NON_ADDABLE_ELEMENTS`: Fe, Zn,
Ni, Cr) nunca entram como adição: o alvo é um *teto* e o teor efetivamente obtido
(``achieved_<el>``) é o que a sucata e o primário trazem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.config import (
    ELEMENT_CO2_KG_PER_KG,
    ELEMENT_PRICE_USD_PER_KG,
    ELEMENT_RECOVERY,
    ELEMENTS,
    NON_ADDABLE_ELEMENTS,
    PRIMARY_AL_CO2_LOW_CARBON,
    PRIMARY_AL_COMPOSITION,
    REMELT_COST_USD_PER_KG,
    SCRAP_GRADES,
    ScrapGrade,
)

CHARGE_COLUMNS: tuple[str, ...] = (
    "scrap_fraction", "primary_fraction", "additions_fraction",
    "cost_USD_per_kg", "CO2_kg_per_kg", "charge_violation_wt",
)



def blend_scrap(scrap_mix: Mapping[str, float],
                grades: Mapping[str, ScrapGrade] = SCRAP_GRADES) -> ScrapGrade:
    """Combina classes de sucata em proporções fixas numa sucata equivalente."""
    total = sum(scrap_mix.values())
    if total <= 0:
        raise ValueError("scrap_mix vazio ou com proporções não positivas.")
    w = {k: v / total for k, v in scrap_mix.items() if v > 0}
    unknown = set(w) - set(grades)
    if unknown:
        raise KeyError(f"Classes de sucata desconhecidas: {sorted(unknown)}")
    comp = {el: sum(w[g] * grades[g].composition.get(el, 0.0) for g in w) for el in ELEMENTS}
    return ScrapGrade(
        name=" + ".join(f"{100 * w[g]:.0f}% {grades[g].name}" for g in w),
        composition=comp,
        price_USD_per_kg=sum(w[g] * grades[g].price_USD_per_kg for g in w),
        co2_kg_per_kg=sum(w[g] * grades[g].co2_kg_per_kg for g in w),
        metal_yield=sum(w[g] * grades[g].metal_yield for g in w),
    )


@dataclass
class ChargeModel:
    """Cálculo de carga vetorizado para uma sucata (ou mistura fixa de sucatas)."""

    scrap: ScrapGrade
    primary_composition: Mapping[str, float] = None  # type: ignore[assignment]
    prices: Mapping[str, float] = None  # type: ignore[assignment]
    co2: Mapping[str, float] = None  # type: ignore[assignment]
    recovery: Mapping[str, float] = None  # type: ignore[assignment]
    low_carbon_primary: bool = False
    non_addable: frozenset[str] = NON_ADDABLE_ELEMENTS

    def __post_init__(self) -> None:
        self.primary_composition = dict(self.primary_composition or PRIMARY_AL_COMPOSITION)
        self.prices = dict(self.prices or ELEMENT_PRICE_USD_PER_KG)
        self.co2 = dict(self.co2 or ELEMENT_CO2_KG_PER_KG)
        self.recovery = dict(self.recovery or ELEMENT_RECOVERY)
        if self.low_carbon_primary:
            self.co2["Al"] = PRIMARY_AL_CO2_LOW_CARBON

    @classmethod
    def from_mix(cls, scrap_mix: Mapping[str, float], low_carbon_primary: bool = False) -> "ChargeModel":
        return cls(blend_scrap(scrap_mix), low_carbon_primary=low_carbon_primary)

    def compute(self, target: pd.DataFrame) -> pd.DataFrame:
        """Carga de custo mínimo para cada composição-alvo (linhas de ``target``).

        Com sucata mais barata que o primário, maximizar a fração de sucata
        minimiza custo e CO2; a fração é limitada pelo elemento cujo excesso
        na sucata exige mais diluição.

        Retorna frações da carga, adições por elemento (``add_<el>``, kg/kg),
        composição obtida (``achieved_<el>``, wt%), custo (USD/kg), CO2 (kg/kg)
        e ``charge_violation_wt`` — o excesso (wt%)
        de impureza que nem a diluição total com primário consegue corrigir
        (> 0 ⇒ alvo inatingível com estas matérias-primas).
        """
        n = len(target)
        els = [el for el in ELEMENTS if el in target]
        ct = np.column_stack([target[el].to_numpy(float) for el in els])          # (n, E)
        cs = np.array([self.scrap.composition.get(el, 0.0) for el in els])       # (E,)
        cp = np.array([self.primary_composition.get(el, 0.0) for el in els])     # (E,)

        na = np.array([el in self.non_addable for el in els])
        # Balanço de massa com as adições só dos elementos adicionáveis; o teor dos não
        # adicionáveis "flutua" (= x_s·c_s + x_p·c_p) e o alvo é apenas um teto.
        T, S, P = ct[:, ~na].sum(1), cs[~na].sum(), cp[~na].sum()
        # x_p(x_s) = p0 + p1·x_s
        p0 = (1.0 - T / 100.0) / (1.0 - P / 100.0)
        p1 = -(1.0 - S / 100.0) / (1.0 - P / 100.0)
        # Adição de cada elemento: a_e(x_s) = (c_t − x_s c_s − x_p c_p)/100 = α_e + β_e·x_s
        alpha = (ct - p0[:, None] * cp) / 100.0
        beta = (-cs - p1 * cp)[None, :] / 100.0 * np.ones((n, 1))

        upper = np.ones(n)
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = np.where(beta < 0, -alpha / beta, np.inf)
            lim = np.where((beta < 0) & (alpha < 0), 0.0, lim)
        upper = np.minimum(upper, lim.min(axis=1))
        if p1 < 0:
            upper = np.minimum(upper, -p0 / p1)
        x_s = np.clip(upper, 0.0, 1.0)

        x_p = p0 + p1 * x_s
        add = alpha + beta * x_s[:, None]
        violation = np.clip(-add, 0.0, None).max(axis=1) * 100.0            # wt% inatingível
        add = np.clip(add, 0.0, None)
        achieved = ct.copy()
        if na.any():  # "adição virtual" ≥ 0 dos não adicionáveis = folga abaixo do teto
            achieved[:, na] = ct[:, na] - 100.0 * add[:, na]
            add[:, na] = 0.0
        x_p = np.clip(x_p, 0.0, None)

        price = np.array([self.prices[el] / self.recovery.get(el, 1.0) for el in els])
        co2 = np.array([self.co2[el] / self.recovery.get(el, 1.0) for el in els])
        y = self.scrap.metal_yield
        rec_al = self.recovery.get("Al", 1.0)
        cost = (x_s * (self.scrap.price_USD_per_kg + REMELT_COST_USD_PER_KG) / y
                + x_p * self.prices["Al"] / rec_al + add @ price)
        carbon = x_s * self.scrap.co2_kg_per_kg / y + x_p * self.co2["Al"] / rec_al + add @ co2

        out = pd.DataFrame(
            {
                "scrap_fraction": x_s,
                "primary_fraction": x_p,
                "additions_fraction": add.sum(axis=1),
                "cost_USD_per_kg": cost,
                "CO2_kg_per_kg": carbon,
                "charge_violation_wt": violation,
            },
            index=target.index,
        )
        for j, el in enumerate(els):
            out[f"add_{el}"] = add[:, j]
        for j, el in enumerate(els):
            out[f"achieved_{el}"] = achieved[:, j]
        return out


@dataclass
class ChargeLPResult:
    success: bool
    fractions: dict[str, float]
    cost_USD_per_kg: float
    CO2_kg_per_kg: float
    message: str = ""

    @property
    def scrap_fraction(self) -> float:
        return sum(v for k, v in self.fractions.items() if k.startswith("scrap:"))


def optimize_charge_lp(
    target: Mapping[str, float],
    grades: Mapping[str, ScrapGrade] = SCRAP_GRADES,
    available_kg: Mapping[str, float] | None = None,
    objective: str = "cost",
    co2_weight: float = 0.0,
    low_carbon_primary: bool = False,
    tolerance_wt: Mapping[str, float] | None = None,
) -> ChargeLPResult:
    """Carga ótima multi-sucata por programação linear (HiGHS).

    Parameters
    ----------
    target:
        Composição-alvo (wt%).
    available_kg:
        Fração máxima (kg por kg de carga) disponível de cada classe de sucata.
    objective:
        ``"cost"`` (USD/kg), ``"co2"`` (kg CO2e/kg) ou ``"weighted"``
        (custo + ``co2_weight``·CO2, ex.: preço de carbono em USD/kg CO2).
    tolerance_wt:
        Faixa ± por elemento (wt%) — especificações de liga são faixas, o que
        aumenta muito a fração de sucata utilizável.

    Elementos não adicionáveis (Fe, Zn, Ni, Cr) só têm teto: ``c ≤ alvo + tol``.
    """
    prices = dict(ELEMENT_PRICE_USD_PER_KG)
    co2 = dict(ELEMENT_CO2_KG_PER_KG)
    if low_carbon_primary:
        co2["Al"] = PRIMARY_AL_CO2_LOW_CARBON
    rec = ELEMENT_RECOVERY
    names = list(grades)
    tol = {el: 0.0 for el in ELEMENTS} | dict(tolerance_wt or {})
    # Variáveis: [x_g (G), x_p, a_e (E)]
    G, E = len(names), len(ELEMENTS)
    cost_vec = np.array(
        [(grades[g].price_USD_per_kg + REMELT_COST_USD_PER_KG) / grades[g].metal_yield for g in names]
        + [prices["Al"] / rec["Al"]] + [prices[e] / rec[e] for e in ELEMENTS])
    co2_vec = np.array(
        [grades[g].co2_kg_per_kg / grades[g].metal_yield for g in names]
        + [co2["Al"] / rec["Al"]] + [co2[e] / rec[e] for e in ELEMENTS])
    c = {"cost": cost_vec, "co2": co2_vec}.get(objective, cost_vec + co2_weight * co2_vec)

    comp_rows = np.zeros((E, G + 1 + E))
    for i, el in enumerate(ELEMENTS):
        comp_rows[i, :G] = [grades[g].composition.get(el, 0.0) for g in names]
        comp_rows[i, G] = PRIMARY_AL_COMPOSITION.get(el, 0.0)
        comp_rows[i, G + 1 + i] = 100.0
    ct = np.array([float(target.get(el, 0.0)) for el in ELEMENTS])
    tol_v = np.array([tol[el] for el in ELEMENTS])
    na = np.array([el in NON_ADDABLE_ELEMENTS for el in ELEMENTS])
    lower = np.where(na, 0.0, ct - tol_v)  # não adicionáveis: apenas teto
    A_ub = np.vstack([comp_rows, -comp_rows])
    b_ub = np.concatenate([ct + tol_v, -lower])
    A_eq = np.ones((1, G + 1 + E))
    bounds = ([(0.0, (available_kg or {}).get(g, 1.0)) for g in names] + [(0.0, 1.0)]
              + [(0.0, 0.0) if el in NON_ADDABLE_ELEMENTS else (0.0, 1.0) for el in ELEMENTS])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if not res.success:
        return ChargeLPResult(False, {}, np.nan, np.nan, res.message)
    x = res.x
    fractions = {f"scrap:{g}": float(x[i]) for i, g in enumerate(names)}
    fractions["primary_Al"] = float(x[G])
    fractions.update({f"add:{e}": float(x[G + 1 + i]) for i, e in enumerate(ELEMENTS)})
    return ChargeLPResult(True, fractions, float(cost_vec @ x), float(co2_vec @ x), res.message)
