"""Design inverso multiobjetivo de ligas de Al secundárias (NSGA-III via pymoo).

Cada candidato é um vetor (composição + processo T6). Para cada um, a PINN
prevê propriedades/fases, o :class:`~src.optimization.charge.ChargeModel`
calcula a carga do forno (sucata + primário + adições) e o
:class:`~src.optimization.thermo_filter.ThermoFilter` avalia o risco.

Objetivos (todos minimizados internamente)
------------------------------------------
1. −UTS              (maximizar resistência)
2. −Alongamento      (maximizar ductilidade)
3. Custo da carga    (USD/kg de metal líquido)
4. Pegada de CO2     (kg CO2e/kg, cradle-to-gate: sucata vs. primário vs. adições)

Restrições (g ≤ 0)
------------------
* UTS ≥ UTS_min, El ≥ El_min, YS ≥ YS_min
* fração de sucata na carga ≥ mínimo (meta de conteúdo reciclado)
* composição atingível com as matérias-primas (sem excesso de impureza)
* nenhuma regra CRÍTICA do filtro termodinâmico e risk_score ≤ máximo
* (opcional) incerteza MC-dropout: σ_UTS/UTS ≤ CV_max

Variáveis em escala log (taxa de resfriamento, tempo de envelhecimento) são
otimizadas em log10.
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
    ELEMENTS,
    INPUT_FEATURES,
    LOG_FEATURES,
    PROCESS_BOUNDS,
    OptimizationConfig,
)
from src.models.pinn import PhysicsInformedNet, predict
from src.optimization.charge import ChargeModel
from src.optimization.thermo_filter import ThermoFilter

logger = logging.getLogger(__name__)

OBJECTIVES: tuple[str, ...] = ("UTS_MPa", "El_pct", "cost_USD_per_kg", "CO2_kg_per_kg")


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
        charge_model: ChargeModel | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.thermo_filter = thermo_filter or ThermoFilter()
        self.charge_model = charge_model or ChargeModel.from_mix(
            config.scrap_mix, low_carbon_primary=config.low_carbon_primary)
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
        n_constr = 7 + int(self.use_uncertainty)
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
        """Predições + carga (custo, CO2, reciclado) + risco para candidatos em unidades físicas."""
        cand = cand.reset_index(drop=True).copy()
        charge = self.charge_model.compute(cand[list(ELEMENTS)]).reset_index(drop=True)
        # Fe/Zn/Ni/Cr: o teor real é o que a carga entrega (alvo = teto)
        for el in self.charge_model.non_addable:
            if el in cand:
                cand[el] = charge[f"achieved_{el}"].to_numpy()
        pred = predict(self.model, cand, n_mc_samples=self.config.n_mc_samples)
        out = pd.concat([cand, pred.reset_index(drop=True), charge], axis=1)
        comp = {e: out[e].to_numpy(float) for e in ELEMENTS}
        out["Si_eq"] = np.asarray(met.silicon_equivalent(comp), float)
        out["sludge_factor"] = np.asarray(met.sludge_factor(comp), float)
        out["Mn_Fe_ratio"] = np.asarray(met.mn_fe_ratio(comp), float)
        out["quality_index_MPa"] = out["UTS_MPa"] + 150.0 * np.log10(out["El_pct"].clip(lower=0.1))
        risk = self.thermo_filter.evaluate_batch(out)
        return pd.concat([out, risk], axis=1)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        res = self.evaluate_candidates(self.decode(X))
        c = self.config
        out["F"] = np.column_stack([
            -res["UTS_MPa"], -res["El_pct"], res["cost_USD_per_kg"], res["CO2_kg_per_kg"],
        ])
        G = [
            (c.min_uts_MPa - res["UTS_MPa"]) / 20.0,
            (c.min_elongation_pct - res["El_pct"]) / 1.0,
            (c.min_ys_MPa - res["YS_MPa"]) / 20.0,
            (c.min_recycled_fraction - res["scrap_fraction"]) / 0.05,
            res["charge_violation_wt"] / 0.05,
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
        charge_model: ChargeModel | None = None,
    ) -> None:
        self.config = config or OptimizationConfig()
        self.problem = AlloyDesignProblem(model, self.config, thermo_filter, bounds, charge_model)

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
        Ex.: ``(0.2, 0.2, 0.1, 0.5)`` privilegia a descarbonização.
        """
        sign = np.array([-1.0 if o in ("UTS_MPa", "El_pct") else 1.0 for o in objectives])
        F = pareto[list(objectives)].to_numpy(float) * sign
        ideal, nadir = F.min(axis=0), F.max(axis=0)
        Fn = (F - ideal) / np.where(nadir - ideal > 1e-12, nadir - ideal, 1.0)
        w = np.asarray(weights, dtype=float)
        w = np.clip(w / w.sum(), 1e-6, None)
        idx = int(ASF().do(Fn, 1.0 / w).argmin())
        return pareto.iloc[idx]
