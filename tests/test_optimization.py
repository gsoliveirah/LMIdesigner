from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ELEMENTS, INPUT_FEATURES, MetallurgicalLimits, OptimizationConfig
from src.optimization.inverse_design import (
    AlloyDesignProblem,
    InverseDesigner,
    alloy_co2,
    alloy_cost,
)
from src.optimization.thermo_filter import Severity, ThermoFilter


def _alloy(**kw: float) -> dict[str, float]:
    base = {el: 0.0 for el in ELEMENTS} | {"C": 0.12, "Mn": 1.8, "Si": 0.3, "N": 0.004,
                                            "T_anneal_C": 800.0, "cooling_rate_Cps": 30.0,
                                            "grain_size_um": 5.0}
    return base | kw


def test_cost_and_co2_increase_with_nickel() -> None:
    lean = pd.DataFrame([_alloy()])
    rich = pd.DataFrame([_alloy(Ni=1.0)])
    assert alloy_cost(rich)[0] > alloy_cost(lean)[0]
    assert alloy_co2(rich)[0] > alloy_co2(lean)[0]
    assert alloy_co2(lean, fe_route="EAF")[0] < alloy_co2(lean)[0]


def test_filter_passes_clean_dp_steel() -> None:
    res = ThermoFilter().evaluate(_alloy())
    assert res.passed
    assert not [f for f in res.flags if f.severity == Severity.CRITICAL]


def test_filter_flags_critical_and_warnings() -> None:
    f = ThermoFilter()
    weld = f.evaluate(_alloy(C=0.35, Mn=3.0, Cr=1.0, Mo=0.5))
    assert not weld.passed
    assert "CEQ_WELDABILITY" in {x.code for x in weld.flags}

    below = f.evaluate(_alloy(T_anneal_C=650.0))
    assert "ANNEAL_BELOW_AC1" in {x.code for x in below.flags}

    boron = f.evaluate(_alloy(B=0.002, Ti=0.0))
    assert "B_UNPROTECTED" in {x.code for x in boron.flags}
    assert f.evaluate(_alloy(B=0.002, Ti=0.03)).risk_score < boron.risk_score

    cu = f.evaluate(_alloy(Cu=0.5, Ni=0.0))
    assert "CU_HOT_SHORTNESS" in {x.code for x in cu.flags}


def test_filter_batch_matches_single() -> None:
    f = ThermoFilter()
    cands = [_alloy(), _alloy(C=0.35, Mn=3.0, Cr=1.0, Mo=0.5), _alloy(Si=1.9)]
    batch = f.evaluate_batch(pd.DataFrame(cands))
    for i, c in enumerate(cands):
        assert batch.loc[i, "risk_score"] == pytest.approx(f.evaluate(c).risk_score)
    assert "SI_GALVANIZING" in batch.loc[2, "flags"]


def test_filter_custom_limits() -> None:
    strict = ThermoFilter(MetallurgicalLimits(ceq_iiw_max=0.30))
    assert not strict.evaluate(_alloy()).passed


def test_problem_decode_log_and_fixed(trained_model) -> None:
    cfg = OptimizationConfig(fixed_values={"Ni": 0.0, "Mo": 0.0})
    prob = AlloyDesignProblem(trained_model, cfg)
    assert "Ni" not in prob.variables and prob.n_var == len(INPUT_FEATURES) - 2
    j = prob.variables.index("cooling_rate_Cps")
    X = (prob.xl + prob.xu)[None, :] / 2
    df = prob.decode(X)
    assert df.loc[0, "cooling_rate_Cps"] == pytest.approx(10.0)  # média geométrica de 1–100
    assert (df["Ni"] == 0).all() and list(df.columns) == list(INPUT_FEATURES)
    out: dict = {}
    prob._evaluate(np.repeat(X, 3, axis=0), out)
    assert out["F"].shape == (3, 4) and out["G"].shape == (3, prob.n_ieq_constr)


def test_inverse_design_run_and_compromise(trained_model) -> None:
    cfg = OptimizationConfig(pop_size=40, n_generations=8, n_partitions=3,
                             min_uts_MPa=600.0, min_elongation_pct=5.0)
    designer = InverseDesigner(trained_model, cfg)
    res = designer.run()
    assert len(res.pareto) > 0
    for col in ("UTS_MPa", "El_pct", "cost_USD_per_kg", "CO2_kg_per_kg", "risk_score"):
        assert col in res.pareto
    if res.feasible:
        assert (res.pareto["UTS_MPa"] >= 600.0 - 1e-6).all()
        assert (res.pareto["n_critical"] == 0).all()
    cheap = InverseDesigner.select_compromise(res.pareto, (0.01, 0.01, 0.97, 0.01))
    strong = InverseDesigner.select_compromise(res.pareto, (0.97, 0.01, 0.01, 0.01))
    assert cheap["cost_USD_per_kg"] <= strong["cost_USD_per_kg"] + 1e-9
    assert strong["UTS_MPa"] >= cheap["UTS_MPa"] - 1e-9
