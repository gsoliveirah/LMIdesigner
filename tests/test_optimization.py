from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ELEMENTS, INPUT_FEATURES, MetallurgicalLimits, OptimizationConfig
from src.optimization.inverse_design import AlloyDesignProblem, InverseDesigner
from src.optimization.thermo_filter import Severity, ThermoFilter


def _alloy(**kw: float) -> dict[str, float]:
    """A356 limpa, T6 de pico, molde permanente."""
    base = {el: 0.0 for el in ELEMENTS} | {
        "Si": 7.0, "Fe": 0.15, "Mn": 0.05, "Mg": 0.35, "Ti": 0.12, "Sr": 0.015,
        "cooling_rate_Cps": 3.0, "T_solution_C": 540.0, "T_aging_C": 170.0, "t_aging_h": 6.0,
    }
    return base | kw


def _codes(result) -> set[str]:
    return {f.code for f in result.flags}


def test_filter_passes_clean_a356() -> None:
    res = ThermoFilter().evaluate(_alloy())
    assert res.passed and res.risk_score == 0.0
    assert not [f for f in res.flags if f.severity == Severity.CRITICAL]


def test_filter_flags_incipient_melting_of_cu_alloy() -> None:
    f = ThermoFilter()
    burnt = f.evaluate(_alloy(Cu=3.5, Mg=0.3, T_solution_C=530.0))
    assert not burnt.passed and "INCIPIENT_MELTING" in _codes(burnt)
    safe = f.evaluate(_alloy(Cu=3.5, Mg=0.3, T_solution_C=495.0))
    assert "INCIPIENT_MELTING" not in _codes(safe)


def test_filter_iron_chemistry_rules() -> None:
    f = ThermoFilter()
    dirty = f.evaluate(_alloy(Fe=0.8, Mn=0.1))
    assert {"FE_ABOVE_CRITICAL", "MN_FE_RATIO_LOW"} <= _codes(dirty)
    neutralized = f.evaluate(_alloy(Fe=0.8, Mn=0.45))
    assert "MN_FE_RATIO_LOW" not in _codes(neutralized)
    assert neutralized.risk_score < dirty.risk_score
    sludge = f.evaluate(_alloy(Fe=1.2, Mn=0.6, Cr=0.15))  # SF = 2.85
    assert not sludge.passed and "SLUDGE_FORMATION" in _codes(sludge)


def test_filter_tramp_and_dissolution_warnings() -> None:
    f = ThermoFilter()
    assert "ZN_TRAMP" in _codes(f.evaluate(_alloy(Zn=1.8)))
    assert "INCOMPLETE_DISSOLUTION" in _codes(f.evaluate(_alloy(Mg=0.6, T_solution_C=490.0)))
    info = f.evaluate(_alloy(Cu=2.0, T_solution_C=500.0))
    assert "CU_CORROSION" in _codes(info) and info.passed


def test_filter_batch_matches_single() -> None:
    f = ThermoFilter()
    cands = [_alloy(), _alloy(Cu=3.5, Mg=0.3, T_solution_C=530.0), _alloy(Fe=0.8), _alloy(Sr=0.04)]
    batch = f.evaluate_batch(pd.DataFrame(cands))
    for i, c in enumerate(cands):
        assert batch.loc[i, "risk_score"] == pytest.approx(f.evaluate(c).risk_score)
    assert "SR_OVERMODIFICATION" in batch.loc[3, "flags"]


def test_filter_custom_limits_and_calphad_columns() -> None:
    strict = ThermoFilter(MetallurgicalLimits(max_freezing_range_C=20.0))
    assert "WIDE_FREEZING_RANGE" in _codes(strict.evaluate(_alloy()))
    res = ThermoFilter().evaluate(_alloy() | {"f_liquid_eq": 0.05, "f_undesired_eq": 0.0})
    assert not res.passed and "INCIPIENT_MELTING_CALPHAD" in _codes(res)


def test_validate_with_calphad_adds_equilibrium_columns() -> None:
    from src.data_pipeline.calphad_gen import EmpiricalThermoBackend

    df = pd.DataFrame([_alloy(), _alloy(Fe=1.2, Mn=0.6, Cr=0.15)])
    out = ThermoFilter(backend=EmpiricalThermoBackend()).validate_with_calphad(df)
    assert {"f_liquid_eq", "f_sludge_eq", "risk_score"} <= set(out)
    assert out.loc[1, "f_sludge_eq"] > 0 and not out.loc[1, "passed"]


def test_problem_decode_log_and_fixed(trained_model) -> None:
    cfg = OptimizationConfig(fixed_values={"Ni": 0.0, "cooling_rate_Cps": 2.0})
    prob = AlloyDesignProblem(trained_model, cfg)
    assert "Ni" not in prob.variables and prob.n_var == len(INPUT_FEATURES) - 2
    X = (prob.xl + prob.xu)[None, :] / 2
    df = prob.decode(X)
    assert df.loc[0, "t_aging_h"] == pytest.approx((0.5 * 24.0) ** 0.5)  # média geométrica
    assert (df["Ni"] == 0).all() and (df["cooling_rate_Cps"] == 2.0).all()
    assert list(df.columns) == list(INPUT_FEATURES)
    out: dict = {}
    prob._evaluate(np.repeat(X, 3, axis=0), out)
    assert out["F"].shape == (3, 4) and out["G"].shape == (3, prob.n_ieq_constr)


def test_candidates_use_achieved_tramp_composition(trained_model) -> None:
    """Zn não é adicionável: com sucata de rodas (Zn 0.03 %), o Zn real fica ≤ alvo."""
    prob = AlloyDesignProblem(trained_model, OptimizationConfig(scrap_mix={"wheels_a356": 1.0}))
    cand = pd.DataFrame([_alloy(Zn=1.5)])[list(INPUT_FEATURES)]
    res = prob.evaluate_candidates(cand)
    assert res.loc[0, "Zn"] < 0.05 and res.loc[0, "add_Zn"] == 0.0


def test_inverse_design_run_and_compromise(trained_model) -> None:
    cfg = OptimizationConfig(pop_size=40, n_generations=8, n_partitions=3,
                             min_uts_MPa=200.0, min_elongation_pct=1.0, scrap_mix={"tense_cast": 1.0})
    res = InverseDesigner(trained_model, cfg).run()
    assert len(res.pareto) > 0
    for col in ("UTS_MPa", "El_pct", "cost_USD_per_kg", "CO2_kg_per_kg", "scrap_fraction", "risk_score"):
        assert col in res.pareto
    if res.feasible:
        assert (res.pareto["UTS_MPa"] >= 200.0 - 1e-6).all()
        assert (res.pareto["n_critical"] == 0).all()
        assert (res.pareto["charge_violation_wt"] <= 1e-9).all()
    green = InverseDesigner.select_compromise(res.pareto, (0.01, 0.01, 0.01, 0.97))
    strong = InverseDesigner.select_compromise(res.pareto, (0.97, 0.01, 0.01, 0.01))
    assert green["CO2_kg_per_kg"] <= strong["CO2_kg_per_kg"] + 1e-9
    assert strong["UTS_MPa"] >= green["UTS_MPa"] - 1e-9


def test_min_recycled_fraction_constraint(trained_model) -> None:
    cfg = OptimizationConfig(pop_size=40, n_generations=10, n_partitions=3, min_uts_MPa=150.0,
                             min_elongation_pct=0.5, min_recycled_fraction=0.7,
                             scrap_mix={"wheels_a356": 1.0})
    res = InverseDesigner(trained_model, cfg).run()
    if res.feasible:
        assert (res.pareto["scrap_fraction"] >= 0.7 - 1e-6).all()
