from __future__ import annotations

import numpy as np
import pytest
import torch

from src import metallurgy as met

A356 = {"Si": 7.0, "Fe": 0.12, "Mg": 0.35, "Ti": 0.12, "Sr": 0.015}


def _sim(comp: dict[str, float], cr: float = 3.0, ts: float = 540.0, ta: float = 170.0, t: float = 6.0):
    c = {k: np.array([v]) for k, v in comp.items()}
    out = met.simulate_casting(c, np.array([cr]), np.array([ts]), np.array([ta]), np.array([t]))
    return {k: float(v[0]) for k, v in out.items()}


def test_silicon_equivalent_and_liquidus() -> None:
    assert met.silicon_equivalent({"Si": 7.0}) == pytest.approx(7.0)
    # Cu abaixa a liquidus ~metade do que o Si por wt%
    assert met.silicon_equivalent({"Si": 7.0, "Cu": 1.0}) == pytest.approx(7.0 + 3.4 / 6.6)
    assert met.liquidus_temperature({}) == pytest.approx(660.452, abs=0.01)
    t_a356 = float(met.liquidus_temperature(A356))
    assert 605.0 < t_a356 < 620.0  # A356: ~613–615 °C


def test_incipient_melting_drops_with_cu_and_mg() -> None:
    assert met.incipient_melting_temperature({"Si": 7}) == pytest.approx(577.0)
    assert met.incipient_melting_temperature({"Cu": 3.5}) == pytest.approx(525.0)
    assert met.incipient_melting_temperature({"Cu": 3.5, "Mg": 0.3}) == pytest.approx(507.0)


def test_iron_chemistry_indices() -> None:
    assert met.sludge_factor({"Fe": 0.8, "Mn": 0.3, "Cr": 0.1}) == pytest.approx(1.7)
    assert met.critical_iron({"Si": 9.0}) == pytest.approx(0.075 * (9.0 - 4.05))
    # Mn neutraliza Fe; resfriamento rápido também favorece α
    lo = float(met.alpha_fe_fraction({"Fe": 0.8, "Mn": 0.1}, 1.0))
    hi = float(met.alpha_fe_fraction({"Fe": 0.8, "Mn": 0.5}, 1.0))
    fast = float(met.alpha_fe_fraction({"Fe": 0.8, "Mn": 0.1}, 50.0))
    assert lo < 0.2 < 0.8 < hi and fast > lo


def test_sdas_power_law() -> None:
    assert float(met.sdas_um(1.0)) == pytest.approx(39.4)
    assert met.sdas_um(10.0) < met.sdas_um(1.0) < met.sdas_um(0.1)


def test_solubility_and_aging_kinetics() -> None:
    assert float(met.cu_solubility(548.0)) == pytest.approx(5.65)
    assert met.cu_solubility(480.0) < met.cu_solubility(520.0)
    assert float(met.aging_response(1.0)) == pytest.approx(1.0)
    assert met.aging_response(0.05) < 1.0 and met.aging_response(50.0) < 1.0
    # Maior temperatura -> cinética mais rápida (P maior para o mesmo tempo)
    assert met.aging_parameter(200.0, 4.0) > met.aging_parameter(160.0, 4.0)


def test_a356_t6_properties_are_realistic() -> None:
    r = _sim(A356)
    assert 180 < r["YS_MPa"] < 240
    assert 260 < r["UTS_MPa"] < 330
    assert 4 < r["El_pct"] < 14
    assert r["YS_MPa"] < r["UTS_MPa"]


def test_iron_harms_and_manganese_mitigates_ductility() -> None:
    clean = _sim(A356, cr=0.3)
    dirty = _sim(A356 | {"Fe": 0.7}, cr=0.3)
    neutralized = _sim(A356 | {"Fe": 0.7, "Mn": 0.4}, cr=0.3)
    assert dirty["El_pct"] < clean["El_pct"]
    assert dirty["El_pct"] < neutralized["El_pct"]
    assert neutralized["f_beta_AlFeSi"] < dirty["f_beta_AlFeSi"]


def test_incipient_melting_destroys_ductility() -> None:
    a319 = {"Si": 6.0, "Fe": 0.4, "Cu": 3.5, "Mn": 0.3, "Mg": 0.3}
    ok = _sim(a319, ts=495.0)
    burnt = _sim(a319, ts=540.0)
    assert burnt["melting_damage"] > 0.99 and ok["melting_damage"] < 0.05
    assert burnt["El_pct"] < 0.5 * ok["El_pct"]


def test_overaging_trades_strength_for_ductility() -> None:
    t6 = _sim(A356, ta=170.0, t=6.0)
    t7 = _sim(A356, ta=220.0, t=8.0)
    assert t7["YS_MPa"] < t6["YS_MPa"] and t7["El_pct"] > t6["El_pct"]


def test_phase_mass_balance_closes() -> None:
    comp = {"Si": 8.0, "Fe": 0.9, "Cu": 3.0, "Mn": 0.3, "Mg": 0.3}
    ph = met.phase_fractions(comp, 1.0, 480.0)
    assert sum(float(v) for v in ph.values()) == pytest.approx(1.0)
    viol = met.mass_balance_residuals(comp, ph)
    assert all(float(v) == pytest.approx(0.0, abs=1e-9) for v in viol.values())
    # Violação detectada: Fe ausente das fases
    bad = dict(ph) | {"f_beta_AlFeSi": 0.0, "f_alpha_AlFeMnSi": 0.0}
    assert float(met.mass_balance_residuals(comp, bad)["Fe"]) > 0.5


def test_functions_are_differentiable_in_torch() -> None:
    comp = {k: torch.tensor([v], requires_grad=True) for k, v in A356.items()}
    cr = torch.tensor([2.0], requires_grad=True)
    ts, ta, t = torch.tensor([540.0]), torch.tensor([170.0]), torch.tensor([6.0])
    ph = met.phase_fractions(comp, cr, ts)
    ys, uts, el = met.mechanical_properties(comp, ph, cr, ts, ta, t)
    (uts + el).sum().backward()
    assert comp["Mg"].grad is not None and torch.isfinite(comp["Mg"].grad).all()
    assert cr.grad is not None and cr.grad.item() > 0  # resfriamento rápido melhora as propriedades
