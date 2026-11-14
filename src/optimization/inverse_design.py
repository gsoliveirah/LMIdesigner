"""Design inverso multiobjetivo de ligas AHSS (NSGA-III via pymoo).

Objetivos (todos minimizados internamente)
------------------------------------------
1. −UTS            (maximizar resistência)
2. −Alongamento    (maximizar ductilidade)
3. Custo de liga   (USD/kg de aço, com rendimento de adição)
4. Pegada de CO2   (kg CO2e/kg de aço, cradle-to-gate)

Restrições (g ≤ 0)
------------------
* UTS ≥ UTS_min, El ≥ El_min
* C_eq(IIW) ≤ C_eq,max
* nenhuma regra CRÍTICA do filtro termodinâmico e risk_score ≤ máximo
* (opcional) incerteza MC-dropout: σ_UTS/UTS ≤ CV_max

Variáveis com escala log (taxa de resfriamento) são otimizadas em log10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.decomposition.asf import ASF
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from src import metallurgy as met
from src.config import (
    COMPOSITION_BOUNDS,
    ELEMENT_CO2_KG_PER_KG,
    ELEMENT_PRICE_USD_PER_KG,
    ELEMENT_RECOVERY,
    ELEMENTS,
    FE_CO2_EAF,
    INPUT_FEATURES,
    LOG_FEATURES,
    PROCESS_BOUNDS,
    OptimizationConfig,
)
from src.models.pinn import PhysicsInformedNet, predict
from src.optimization.thermo_filter import ThermoFilter

logger = logging.getLogger(__name__)

OBJECTIVES: tuple[str, ...] = ("UTS_MPa", "El_pct", "cost_USD_per_kg", "CO2_kg_per_kg")


# ---------------------------------------------------------------------------
# Modelos de custo e emissão
# ---------------------------------------------------------------------------
def alloy_cost(
    df: pd.DataFrame,
    prices: Mapping[str, float] = ELEMENT_PRICE_USD_PER_KG,
    recovery: Mapping[str, float] = ELEMENT_RECOVERY,
) -> np.ndarray:
    """Custo de matéria-prima (USD/kg de aço) com rendimento de adição."""
    alloy_wt = sum(df[el].to_numpy(float) for el in ELEMENTS if el in df)
    cost = (100.0 - alloy_wt) / 100.0 * prices["Fe"]
    for el in ELEMENTS:
        if el in df:
            cost = cost + df[el].to_numpy(float) / 100.0 * prices[el] / recovery.get(el, 1.0)
    return np.asarray(cost, dtype=float)


def alloy_co2(
    df: pd.DataFrame,
    factors: Mapping[str, float] = ELEMENT_CO2_KG_PER_KG,
    recovery: Mapping[str, float] = ELEMENT_RECOVERY,
    fe_route: str = "BF-BOF",
) -> np.ndarray:
    """Pegada de carbono (kg CO2e/kg de aço) das matérias-primas."""
    fe_factor = FE_CO2_EAF if fe_route.upper() == "EAF" else factors["Fe"]
    alloy_wt = sum(df[el].to_numpy(float) for el in ELEMENTS if el in df)
    co2 = (100.0 - alloy_wt) / 100.0 * fe_factor
    for el in ELEMENTS:
        if el in df:
            co2 = co2 + df[el].to_numpy(float) / 100.0 * factors[el] / recovery.get(el, 1.0)
    return np.asarray(co2, dtype=float)


# ---------------------------------------------------------------------------
# Problema pymoo
# ---------------------------------------------------------------------------
class AlloyDesignProblem(Problem):
    """Problema vetorizado: cada linha de X é um candidato (liga + processo)."""

    def __init__(
        self,
        model: PhysicsInformedNet,
        config: OptimizationConfig,
        thermo_filter: ThermoFilter | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.thermo_filter = thermo_filter or ThermoFilter()
        all_bounds = dict(bounds) if bounds is not None else {**COMPOSITION_BOUNDS, **PROCESS_BOUNDS}
        self.fixed: dict[str, float] = {k: float(v) for k, v in config.fixed_values.items()}
        for name, (lo, hi) in all_bounds.items():
            if hi <= lo and name not in self.fixed:
                self.fixed[name] = float(lo)
        self.variables: list[str] = [f for f in INPUT_FEATURES if f not in self.fixed]
        xl, xu = [], []
        for name in self.variables:
            lo, hi = all_bounds[name]
            if name in LOG_FEATURES:
                lo, hi = np.log10(lo), np.log10(hi)
            xl.append(lo)
            xu.append(hi)
        self.use_uncertainty = config.max_uts_cv is not None and config.n_mc_samples > 0
        n_constr = 5 + int(self.use_uncertainty)
        super().__init__(n_var=len(self.variables), n_obj=4, n_ieq_constr=n_constr,
                         xl=np.array(xl), xu=np.array(xu))

    def decode(self, X: np.ndarray) -> pd.DataFrame:
        """Vetor de decisão -> DataFrame com todas as features em unidades físicas."""
        X = np.atleast_2d(X)
        df = pd.DataFrame(index=range(len(X)))
        for j, name in enumerate(self.variables):
            df[name] = 10 ** X[:, j] if name in LOG_FEATURES else X[:, j]
        for name, value in self.fixed.items():
            df[name] = value
        return df[list(INPUT_FEATURES)]

    def evaluate_candidates(self, cand: pd.DataFrame) -> pd.DataFrame:
        """Predições + custo + CO2 + risco para candidatos em unidades físicas."""
        pred = predict(self.model, cand, n_mc_samples=self.config.n_mc_samples)
        out = pd.concat([cand.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        out["cost_USD_per_kg"] = alloy_cost(out)
        out["CO2_kg_per_kg"] = alloy_co2(out, fe_route=self.config.fe_route)
        out["Ceq_IIW"] = np.asarray(met.carbon_equivalent_iiw({e: out[e] for e in ELEMENTS}), float)
        out["UTSxEl_GPa_pct"] = out["UTS_MPa"] * out["El_pct"] / 1000.0
        risk = self.thermo_filter.evaluate_batch(out)
        return pd.concat([out, risk], axis=1)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        res = self.evaluate_candidates(self.decode(X))
        c = self.config
        out["F"] = np.column_stack([
            -res["UTS_MPa"], -res["El_pct"], res["cost_USD_per_kg"], res["CO2_kg_per_kg"],
        ])
        G = [
            (c.min_uts_MPa - res["UTS_MPa"]) / 100.0,
            (c.min_elongation_pct - res["El_pct"]) / 5.0,
            (res["Ceq_IIW"] - c.ceq_max) / 0.1,
            res["n_critical"].to_numpy(float) - 0.5,
            res["risk_score"] - c.max_risk_score,
        ]
        if self.use_uncertainty:
            G.append(res["UTS_MPa_std"] / res["UTS_MPa"].clip(lower=1.0) - c.max_uts_cv)
        out["G"] = np.column_stack(G)


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------
@dataclass
class DesignResult:
    pareto: pd.DataFrame
    n_evaluations: int
    feasible: bool
    history: list[dict[str, float]] = field(default_factory=list)


class InverseDesigner:
    """Design inverso: encontra ligas na fronteira de Pareto para os alvos dados."""

    def __init__(
        self,
        model: PhysicsInformedNet,
        config: OptimizationConfig | None = None,
        thermo_filter: ThermoFilter | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.problem = AlloyDesignProblem(model, self.config, thermo_filter, bounds)

    def run(self, verbose: bool = False) -> DesignResult:
        c = self.config
        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=c.n_partitions)
        pop = max(c.pop_size, len(ref_dirs) + (4 - len(ref_dirs) % 4) % 4)
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop)
        res = minimize(self.problem, algorithm, ("n_gen", c.n_generations),
                       seed=c.seed, verbose=verbose, save_history=False)

        feasible = res.X is not None
        if feasible:
            X = np.atleast_2d(res.X)
        else:  # nenhuma solução viável: devolve a menos inviável da população
            logger.warning("Nenhuma solução viável encontrada; retornando menor violação.")
            cv = res.pop.get("CV").ravel()
            X = res.pop.get("X")[np.argsort(cv)[:10]]
        pareto = self.problem.evaluate_candidates(self.problem.decode(X))
        pareto = pareto.sort_values("UTS_MPa", ascending=False).reset_index(drop=True)
        pareto["feasible"] = feasible
        return DesignResult(pareto=pareto, n_evaluations=int(res.algorithm.evaluator.n_eval),
                            feasible=feasible)

    @staticmethod
    def select_compromise(
        pareto: pd.DataFrame,
        weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        objectives: Sequence[str] = OBJECTIVES,
    ) -> pd.Series:
        """Seleciona a solução de compromisso por Achievement Scalarizing Function.

        ``weights`` expressa a importância relativa de cada objetivo (UTS, El,
        custo, CO2); pesos maiores puxam a escolha para aquele objetivo.
        """
        sign = np.array([-1.0 if o in ("UTS_MPa", "El_pct") else 1.0 for o in objectives])
        F = pareto[list(objectives)].to_numpy(float) * sign
        ideal, nadir = F.min(axis=0), F.max(axis=0)
        Fn = (F - ideal) / np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        w = np.asarray(weights, dtype=float)
        w = np.clip(w / w.sum(), 1e-6, None)
        idx = int(ASF().do(Fn, 1.0 / w).argmin())
        return pareto.iloc[idx]
