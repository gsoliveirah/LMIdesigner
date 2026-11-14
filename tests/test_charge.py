from __future__ import annotations

import pandas as pd
import pytest

from src.config import ELEMENTS, SCRAP_GRADES
from src.optimization.charge import ChargeModel, blend_scrap, optimize_charge_lp

A356_TARGET = {"Si": 7.0, "Fe": 0.15, "Cu": 0.02, "Mn": 0.02, "Mg": 0.35, "Zn": 0.03,
               "Cr": 0.0, "Ni": 0.0, "Ti": 0.12, "Sr": 0.01}


def _frame(*targets: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{el: t.get(el, 0.0) for el in ELEMENTS} for t in targets])


def test_mass_balance_reproduces_target() -> None:
    target = A356_TARGET | {"Fe": 0.4, "Cu": 0.8, "Zn": 0.6, "Ni": 0.05, "Cr": 0.03}
    cm = ChargeModel.from_mix({"tense_cast": 1.0})
    ch = cm.compute(_frame(target)).iloc[0]
    total = ch["scrap_fraction"] + ch["primary_fraction"] + ch["additions_fraction"]
    assert total == pytest.approx(1.0, abs=1e-9)
    for el in ELEMENTS:
        made = (ch["scrap_fraction"] * cm.scrap.composition.get(el, 0.0)
                + ch["primary_fraction"] * cm.primary_composition.get(el, 0.0) + 100 * ch[f"add_{el}"])
        assert made == pytest.approx(ch[f"achieved_{el}"], abs=1e-9)
    assert ch["charge_violation_wt"] == 0.0


def test_scrap_fraction_limited_by_tramp_element() -> None:
    cm = ChargeModel.from_mix({"tense_cast": 1.0})  # Fe 0.95, Zn 1.2, Cu 2.5
    loose = cm.compute(_frame(A356_TARGET | {"Fe": 0.8, "Cu": 2.5, "Zn": 1.2, "Ni": 0.1, "Cr": 0.05,
                                             "Mn": 0.3, "Si": 8.5})).iloc[0]
    tight = cm.compute(_frame(A356_TARGET | {"Fe": 0.3, "Cu": 2.5, "Zn": 1.2, "Ni": 0.1, "Cr": 0.05,
                                             "Mn": 0.3, "Si": 8.5})).iloc[0]
    assert tight["scrap_fraction"] < loose["scrap_fraction"]
    # Fe é o limitante: atinge exatamente o alvo sem nenhuma adição
    assert tight["achieved_Fe"] == pytest.approx(0.3) and tight["add_Fe"] == pytest.approx(0.0, abs=1e-12)
    assert 0.2 < tight["scrap_fraction"] < 0.3
    assert tight["CO2_kg_per_kg"] > loose["CO2_kg_per_kg"]


def test_low_carbon_primary_reduces_co2() -> None:
    target = _frame(A356_TARGET)
    std = ChargeModel.from_mix({"tense_cast": 1.0}).compute(target).iloc[0]
    hydro = ChargeModel.from_mix({"tense_cast": 1.0}, low_carbon_primary=True).compute(target).iloc[0]
    assert hydro["CO2_kg_per_kg"] < std["CO2_kg_per_kg"]


def test_unreachable_target_reports_violation() -> None:
    # Fe abaixo do Fe do próprio Al primário (0.10 %): inatingível
    ch = ChargeModel.from_mix({"wheels_a356": 1.0}).compute(_frame(A356_TARGET | {"Fe": 0.05})).iloc[0]
    assert ch["charge_violation_wt"] > 0.0


def test_non_addable_zinc_is_capped_not_added() -> None:
    ch = ChargeModel.from_mix({"wheels_a356": 1.0}).compute(_frame(A356_TARGET | {"Zn": 1.0})).iloc[0]
    assert ch["add_Zn"] == 0.0
    assert ch["achieved_Zn"] < 0.05
    assert ch["charge_violation_wt"] == 0.0


def test_blend_scrap_is_weighted_average() -> None:
    mix = blend_scrap({"tense_cast": 0.5, "wheels_a356": 0.5})
    expected = 0.5 * SCRAP_GRADES["tense_cast"].composition["Fe"] + 0.5 * SCRAP_GRADES["wheels_a356"].composition["Fe"]
    assert mix.composition["Fe"] == pytest.approx(expected)
    with pytest.raises(KeyError):
        blend_scrap({"unknown": 1.0})


def test_lp_matches_closed_form_for_single_grade() -> None:
    target = A356_TARGET | {"Fe": 0.4, "Cu": 1.0, "Zn": 0.5, "Ni": 0.05, "Cr": 0.03, "Mn": 0.2}
    closed = ChargeModel.from_mix({"tense_cast": 1.0}).compute(_frame(target)).iloc[0]
    lp = optimize_charge_lp(target, grades={"tense_cast": SCRAP_GRADES["tense_cast"]})
    assert lp.success
    assert lp.scrap_fraction == pytest.approx(closed["scrap_fraction"], abs=1e-6)
    assert lp.cost_USD_per_kg == pytest.approx(closed["cost_USD_per_kg"], abs=1e-6)


def test_lp_multi_grade_beats_single_and_respects_availability() -> None:
    single = optimize_charge_lp(A356_TARGET, grades={"tense_cast": SCRAP_GRADES["tense_cast"]})
    multi = optimize_charge_lp(A356_TARGET)
    assert multi.success and multi.cost_USD_per_kg <= single.cost_USD_per_kg + 1e-9
    assert multi.fractions["scrap:wheels_a356"] > 0.5  # sucata limpa é a escolha natural para A356
    capped = optimize_charge_lp(A356_TARGET, available_kg={"wheels_a356": 0.2})
    assert capped.fractions["scrap:wheels_a356"] <= 0.2 + 1e-9
    assert capped.CO2_kg_per_kg > multi.CO2_kg_per_kg


def test_lp_co2_objective_and_tolerance() -> None:
    target = A356_TARGET | {"Fe": 0.3}
    cost_opt = optimize_charge_lp(target, objective="cost")
    co2_opt = optimize_charge_lp(target, objective="co2")
    assert co2_opt.CO2_kg_per_kg <= cost_opt.CO2_kg_per_kg + 1e-9
    relaxed = optimize_charge_lp(target, tolerance_wt={"Fe": 0.1, "Cu": 0.1, "Mn": 0.1})
    assert relaxed.cost_USD_per_kg <= cost_opt.cost_USD_per_kg + 1e-9
