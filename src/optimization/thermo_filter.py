"""Filtro de estabilidade de fases e risco metalúrgico para ligas de Al secundárias.

Duas camadas:

1. **Triagem rápida vetorizada** (:meth:`ThermoFilter.evaluate_batch`) —
   regras metalúrgicas usadas *dentro* do laço do NSGA-III (milhares de
   avaliações por geração): química do Fe (Fe crítico, Mn/Fe, sludge),
   fusão incipiente na solubilização, intervalo de solidificação, impurezas
   acumulativas (Zn, Ni), modificadores e refinadores.
2. **Validação CALPHAD** (:meth:`ThermoFilter.validate_with_calphad`) —
   equilíbrio completo apenas no conjunto de Pareto final: líquido na T de
   solubilização e intermetálicos primários (sludge) na T do forno de espera.

Cada regra produz um *excesso relativo* ≥ 0; o ``risk_score`` agrega os
excessos ponderados pela severidade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src import metallurgy as met
from src.config import ELEMENTS, MetallurgicalLimits
from src.data_pipeline.calphad_gen import ThermoBackend


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.INFO: 0.0, Severity.WARNING: 0.25, Severity.CRITICAL: 1.0,
}


@dataclass(frozen=True)
class RiskFlag:
    code: str
    severity: Severity
    message: str
    value: float
    limit: float


@dataclass
class FilterResult:
    passed: bool
    risk_score: float
    flags: list[RiskFlag] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """Regra vetorizada: ``value(df) > limit`` (ou ``<`` se ``lower``)."""

    code: str
    severity: Severity
    message: str
    value: Callable[[pd.DataFrame], np.ndarray]
    limit: Callable[[pd.DataFrame], np.ndarray | float]
    lower: bool = False  # True -> violação quando value < limit
    applies: Callable[[pd.DataFrame], np.ndarray] | None = None

    def excess(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(self.value(df), dtype=float)
        lim = np.broadcast_to(np.asarray(self.limit(df), dtype=float), v.shape)
        scale = np.maximum(np.abs(lim), 1e-9)
        exc = (lim - v) / scale if self.lower else (v - lim) / scale
        exc = np.clip(exc, 0.0, None)
        if self.applies is not None:
            exc = np.where(np.asarray(self.applies(df), dtype=bool), exc, 0.0)
        return v, exc


def _col(name: str, default: float = 0.0) -> Callable[[pd.DataFrame], np.ndarray]:
    def f(df: pd.DataFrame) -> np.ndarray:
        return df[name].to_numpy(float) if name in df else np.full(len(df), default)
    return f


def _comp(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {el: df[el].to_numpy(float) if el in df else np.zeros(len(df)) for el in ELEMENTS}


def _dissolution_ratio(df: pd.DataFrame) -> np.ndarray:
    """(Cu_d + Mg_d)/(Cu + Mg) na T de solubilização."""
    comp = _comp(df)
    T = _col("T_solution_C", 540.0)(df)
    cu_d, mg_d = met.dissolved_solutes(comp, T)
    total = comp["Cu"] + comp["Mg"]
    return np.where(total > 1e-6, (np.asarray(cu_d) + np.asarray(mg_d)) / np.maximum(total, 1e-6), 1.0)


class ThermoFilter:
    """Avalia estabilidade de fases e riscos de processamento/aplicação."""

    def __init__(
        self,
        limits: MetallurgicalLimits | None = None,
        backend: ThermoBackend | None = None,
    ) -> None:
        self.limits = limits or MetallurgicalLimits()
        self.backend = backend
        self.rules = self._build_rules()

    # ------------------------------------------------------------------ regras
    def _build_rules(self) -> list[Rule]:
        L = self.limits
        t_inc = lambda df: np.asarray(met.incipient_melting_temperature(_comp(df)), float)  # noqa: E731
        has = lambda col: (lambda df: np.full(len(df), col in df))  # noqa: E731
        return [
            Rule("INCIPIENT_MELTING", Severity.CRITICAL,
                 "Solubilização acima (ou a < margem) do eutético de menor ponto de fusão: "
                 "fusão incipiente/\"queima\" de contornos (Al2Cu, fase Q).",
                 _col("T_solution_C", -np.inf),
                 lambda df: t_inc(df) - L.incipient_melting_margin_C, applies=has("T_solution_C")),
            Rule("SLUDGE_FORMATION", Severity.CRITICAL,
                 "Sludge factor (Fe+2Mn+3Cr) acima do crítico na T de espera: "
                 "intermetálicos primários sedimentam no forno (perda de Fe/Mn, inclusões duras).",
                 lambda df: np.asarray(met.sludge_factor(_comp(df)), float),
                 lambda df: np.asarray(met.critical_sludge_factor(
                     L.holding_temperature_C, L.sludge_factor_base, L.sludge_factor_slope_per_C), float)),
            Rule("FE_ABOVE_CRITICAL", Severity.WARNING,
                 "Fe acima do Fe crítico (Taylor): β-Al5FeSi pré-eutético grosseiro, "
                 "porosidade de contração e perda de ductilidade.",
                 _col("Fe"), lambda df: np.asarray(met.critical_iron(_comp(df)), float)),
            Rule("MN_FE_RATIO_LOW", Severity.WARNING,
                 "Razão (Mn+2Cr)/Fe < 0.5 com Fe elevado: Fe não neutralizado (β em agulhas).",
                 lambda df: np.asarray(met.mn_fe_ratio(_comp(df)), float),
                 lambda df: L.min_mn_fe_ratio, lower=True,
                 applies=lambda df: _col("Fe")(df) > L.mn_fe_ratio_applies_above_fe),
            Rule("BETA_PHASE_FRACTION", Severity.WARNING,
                 "Fração prevista de β-Al5FeSi elevada.",
                 _col("f_beta_AlFeSi"), lambda df: L.max_beta_fraction, applies=has("f_beta_AlFeSi")),
            Rule("INCOMPLETE_DISSOLUTION", Severity.WARNING,
                 "Solubilização insuficiente: Cu/Mg não dissolvidos não contribuem ao endurecimento T6.",
                 lambda df: _dissolution_ratio(df), lambda df: L.min_dissolution_ratio, lower=True,
                 applies=has("T_solution_C")),
            Rule("WIDE_FREEZING_RANGE", Severity.WARNING,
                 "Intervalo de solidificação largo: porosidade de contração e trinca a quente.",
                 lambda df: np.asarray(met.freezing_range(_comp(df)), float),
                 lambda df: L.max_freezing_range_C),
            Rule("ZN_TRAMP", Severity.WARNING,
                 "Zn acumulado da sucata acima do limite (trinca a quente, corrosão, normas EN AC).",
                 _col("Zn"), lambda df: L.max_zn),
            Rule("NI_TRAMP", Severity.WARNING,
                 "Ni acumulado acima do limite (intermetálicos Al3Ni, ductilidade).",
                 _col("Ni"), lambda df: L.max_ni),
            Rule("CU_CORROSION", Severity.INFO,
                 "Cu > 1 %: resistência à corrosão reduzida (aceitável para A380/319).",
                 _col("Cu"), lambda df: L.max_cu_corrosion),
            Rule("SR_OVERMODIFICATION", Severity.WARNING,
                 "Sr em excesso: sobremodificação, Al2SrSi2 e porosidade gasosa.",
                 _col("Sr"), lambda df: L.max_sr),
            Rule("TI_PRIMARY_TIAL3", Severity.WARNING,
                 "Ti em excesso: TiAl3 primário (inclusões duras, sedimentação).",
                 _col("Ti"), lambda df: L.max_ti),
            Rule("HYPEREUTECTIC", Severity.WARNING,
                 "Si equivalente próximo/acima do eutético: Si primário, fora da validade do modelo.",
                 lambda df: np.asarray(met.silicon_equivalent(_comp(df)), float),
                 lambda df: L.max_si_hypoeutectic),
            Rule("INCIPIENT_MELTING_CALPHAD", Severity.CRITICAL,
                 "CALPHAD prevê líquido na temperatura de solubilização.",
                 _col("f_liquid_eq"), lambda df: 1e-3, applies=has("f_liquid_eq")),
            Rule("UNDESIRED_PHASES_CALPHAD", Severity.CRITICAL,
                 "CALPHAD prevê intermetálicos primários (sludge) no forno de espera.",
                 _col("f_undesired_eq"), lambda df: L.max_undesired_phase_fraction,
                 applies=has("f_undesired_eq")),
        ]

    # ------------------------------------------------------------------ avaliação
    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Avaliação vetorizada.

        Retorna DataFrame com ``risk_score``, ``n_critical``, ``n_warning``,
        ``passed``, ``flags`` (códigos separados por ``;``) e o excesso de cada regra.
        """
        out = pd.DataFrame(index=df.index)
        risk = np.zeros(len(df))
        n_crit = np.zeros(len(df), dtype=int)
        n_warn = np.zeros(len(df), dtype=int)
        codes: list[list[str]] = [[] for _ in range(len(df))]
        for rule in self.rules:
            _, exc = rule.excess(df)
            violated = exc > 0
            out[f"risk_{rule.code}"] = exc
            risk += np.where(violated, SEVERITY_WEIGHT[rule.severity] * (1.0 + np.minimum(exc, 1.0)), 0.0)
            if rule.severity == Severity.CRITICAL:
                n_crit += violated
            elif rule.severity == Severity.WARNING:
                n_warn += violated
            for i in np.flatnonzero(violated):
                codes[i].append(rule.code)
        out["risk_score"] = risk
        out["n_critical"] = n_crit
        out["n_warning"] = n_warn
        out["passed"] = n_crit == 0
        out["flags"] = [";".join(c) for c in codes]
        return out

    def evaluate(self, candidate: Mapping[str, float]) -> FilterResult:
        """Avaliação detalhada de um único candidato."""
        df = pd.DataFrame([dict(candidate)])
        flags: list[RiskFlag] = []
        for rule in self.rules:
            value, exc = rule.excess(df)
            if exc[0] > 0:
                lim = float(np.asarray(rule.limit(df), dtype=float).ravel()[0])
                flags.append(RiskFlag(rule.code, rule.severity, rule.message, float(value[0]), lim))
        batch = self.evaluate_batch(df).iloc[0]
        comp = _comp(df)

        def _f(x: object) -> float:
            return float(np.asarray(x, dtype=float).ravel()[0])

        metrics = {
            "Si_eq": _f(met.silicon_equivalent(comp)),
            "T_liquidus_C": _f(met.liquidus_temperature(comp)),
            "T_incipient_C": _f(met.incipient_melting_temperature(comp)),
            "freezing_range_C": _f(met.freezing_range(comp)),
            "sludge_factor": _f(met.sludge_factor(comp)),
            "Mn_Fe_ratio": _f(met.mn_fe_ratio(comp)),
            "Fe_critical": _f(met.critical_iron(comp)),
        }
        return FilterResult(passed=bool(batch["passed"]), risk_score=float(batch["risk_score"]),
                            flags=flags, metrics=metrics)

    def validate_with_calphad(self, df: pd.DataFrame, backend: ThermoBackend | None = None) -> pd.DataFrame:
        """Recalcula o equilíbrio (solubilização + forno de espera) e reavalia os riscos.

        Adiciona as colunas de :data:`EQ_COLUMNS` (``f_liquid_eq``, ``f_sludge_eq``, …).
        """
        backend = backend or self.backend
        if backend is None:
            raise ValueError("Nenhum backend termodinâmico configurado.")
        eq = backend.equilibrium_batch(df[list(ELEMENTS)], df["T_solution_C"].to_numpy(float))
        merged = pd.concat([df.drop(columns=[c for c in eq.columns if c in df]), eq], axis=1)
        risk = self.evaluate_batch(merged)
        return pd.concat([merged.drop(columns=[c for c in risk.columns if c in merged]), risk], axis=1)
