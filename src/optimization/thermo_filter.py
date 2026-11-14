"""Filtro de estabilidade de fases e risco metalúrgico.

Duas camadas:

1. **Triagem rápida vetorizada** (:meth:`ThermoFilter.evaluate_batch`) —
   regras metalúrgicas empíricas usadas *dentro* do laço do NSGA-III
   (milhares de avaliações por geração).
2. **Validação CALPHAD** (:meth:`ThermoFilter.validate_with_calphad`) —
   equilíbrio completo apenas no conjunto de Pareto final, verificando fases
   indesejadas (σ, Laves, µ…) e a fração de austenita intercrítica.

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
        ac1 = lambda df: np.asarray(met.ac1_andrews(_comp(df)), float)  # noqa: E731
        ac3 = lambda df: np.asarray(met.ac3_andrews(_comp(df)), float)  # noqa: E731
        has_T = lambda df: np.full(len(df), "T_anneal_C" in df)  # noqa: E731
        return [
            Rule("CEQ_WELDABILITY", Severity.CRITICAL,
                 "Carbono equivalente IIW acima do limite de soldabilidade (soldagem a ponto/HAZ).",
                 lambda df: np.asarray(met.carbon_equivalent_iiw(_comp(df)), float),
                 lambda df: L.ceq_iiw_max),
            Rule("PCM_COLD_CRACKING", Severity.WARNING,
                 "Pcm elevado: risco de trinca a frio induzida por hidrogênio.",
                 lambda df: np.asarray(met.pcm_ito_bessyo(_comp(df)), float),
                 lambda df: L.pcm_max,
                 applies=lambda df: _col("C")(df) <= 0.18),  # domínio de validade do Pcm
            Rule("NARROW_INTERCRITICAL_WINDOW", Severity.WARNING,
                 "Janela intercrítica (Ac3−Ac1) estreita: baixa robustez de processo na linha de recozimento.",
                 lambda df: ac3(df) - ac1(df), lambda df: L.min_intercritical_window_C, lower=True),
            Rule("ANNEAL_BELOW_AC1", Severity.CRITICAL,
                 "Temperatura de recozimento abaixo de Ac1: não há austenita para formar fases duras.",
                 _col("T_anneal_C", np.inf), ac1, lower=True, applies=has_T),
            Rule("ANNEAL_GRAIN_COARSENING", Severity.WARNING,
                 "Recozimento > Ac3 + 50 °C: crescimento de grão austenítico.",
                 _col("T_anneal_C", -np.inf), lambda df: ac3(df) + 50.0, applies=has_T),
            Rule("MN_BANDING", Severity.WARNING,
                 "Mn elevado: segregação/bandeamento e risco de fragilização por H.",
                 _col("Mn"), lambda df: L.max_mn_banding),
            Rule("SI_GALVANIZING", Severity.WARNING,
                 "Si elevado: óxidos superficiais prejudicam a galvanização por imersão.",
                 _col("Si"), lambda df: L.max_si_galvanizing),
            Rule("AL_CASTING", Severity.WARNING,
                 "Al elevado: entupimento de válvula no lingotamento contínuo / AlN.",
                 _col("Al"), lambda df: L.max_al_casting),
            Rule("B_EMBRITTLEMENT", Severity.CRITICAL,
                 "B acima do limite: boretos em contorno de grão (fragilização).",
                 _col("B"), lambda df: L.max_b_embrittlement),
            Rule("B_UNPROTECTED", Severity.WARNING,
                 "B ativo sem Ti suficiente (Ti/N < 3.42): formação de BN anula a temperabilidade.",
                 lambda df: _col("Ti")(df) / np.maximum(_col("N")(df), 1e-6),
                 lambda df: L.min_ti_n_ratio, lower=True,
                 applies=lambda df: _col("B")(df) > 0.0005),
            Rule("MICROALLOY_EXCESS", Severity.WARNING,
                 "Nb+Ti+V elevado: carbonitretos grosseiros, trincas no lingotamento.",
                 lambda df: _col("Nb")(df) + _col("Ti")(df) + _col("V")(df),
                 lambda df: L.max_microalloy_sum),
            Rule("CU_HOT_SHORTNESS", Severity.WARNING,
                 "Cu sem Ni suficiente: fragilidade a quente (hot shortness).",
                 lambda df: _col("Cu")(df) / np.maximum(_col("Ni")(df), 1e-3),
                 lambda df: L.cu_ni_ratio_hot_shortness,
                 applies=lambda df: _col("Cu")(df) > 0.25),
            Rule("UNDESIRED_PHASES_CALPHAD", Severity.CRITICAL,
                 "CALPHAD prevê fases intermetálicas indesejadas (σ, Laves, µ…).",
                 _col("f_undesired_eq"), lambda df: L.max_undesired_phase_fraction),
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
        metrics = {
            "Ceq_IIW": float(np.asarray(met.carbon_equivalent_iiw(comp))[0]),
            "Pcm": float(np.asarray(met.pcm_ito_bessyo(comp))[0]),
            "Ac1_C": float(np.asarray(met.ac1_andrews(comp))[0]),
            "Ac3_C": float(np.asarray(met.ac3_andrews(comp))[0]),
            "Ms_C": float(np.asarray(met.ms_andrews(comp))[0]),
        }
        return FilterResult(passed=bool(batch["passed"]), risk_score=float(batch["risk_score"]),
                            flags=flags, metrics=metrics)

    def validate_with_calphad(self, df: pd.DataFrame, backend: ThermoBackend | None = None) -> pd.DataFrame:
        """Recalcula o equilíbrio na T de recozimento e reavalia os riscos.

        Adiciona ``f_austenite_eq``, ``C_austenite_eq_wt`` e ``f_undesired_eq``.
        """
        backend = backend or self.backend
        if backend is None:
            raise ValueError("Nenhum backend termodinâmico configurado.")
        eq = backend.equilibrium_batch(df[list(ELEMENTS)], df["T_anneal_C"].to_numpy(float))
        merged = pd.concat([df.drop(columns=[c for c in eq.columns if c in df]), eq], axis=1)
        risk = self.evaluate_batch(merged)
        return pd.concat([merged.drop(columns=[c for c in risk.columns if c in merged]), risk], axis=1)
