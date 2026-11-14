from __future__ import annotations

import numpy as np
import pytest
import torch

from src import metallurgy as met


def test_carbon_equivalent_iiw_known_value() -> None:
    comp = {"C": 0.10, "Mn": 1.2, "Cr": 0.5, "Mo": 0.2, "V": 0.05, "Ni": 0.3, "Cu": 0.15}
    expected = 0.10 + 1.2 / 6 + (0.5 + 0.2 + 0.05) / 5 + (0.3 + 0.15) / 15
    assert met.carbon_equivalent_iiw(comp) == pytest.approx(expected)


def test_critical_temperatures_plain_carbon() -> None:
    comp = {"C": 0.2, "Mn": 0.0}
    assert met.ac1_andrews(comp) == pytest.approx(723.0)
    assert met.ac3_andrews(comp) == pytest.approx(910 - 203 * 0.2**0.5)
    assert met.ac3_andrews(comp) > met.ac1_andrews(comp)


def test_ms_decreases_with_carbon_and_mn() -> None:
    assert met.ms_andrews({"C": 0.4}) < met.ms_andrews({"C": 0.1})
    assert met.ms_andrews({"C": 0.1, "Mn": 3}) < met.ms_andrews({"C": 0.1, "Mn": 1})


def test_koistinen_marburger_bounds() -> None:
    assert met.koistinen_marburger(np.array(20.0)) == pytest.approx(0.0)
    assert 0.98 < float(met.koistinen_marburger(np.array(400.0))) < 1.0


def test_hall_petch_finer_grain_is_stronger() -> None:
    comp = {"Mn": 1.5, "Si": 0.3, "N": 0.005}
    assert met.ferrite_yield_strength(comp, 3.0) > met.ferrite_yield_strength(comp, 15.0)


def test_microstructure_fractions_sum_to_one() -> None:
    n = 50
    rng = np.random.default_rng(0)
    comp = {"C": rng.uniform(0.05, 0.4, n), "Mn": rng.uniform(0.5, 3.5, n),
            "Si": rng.uniform(0, 2, n), "Al": rng.uniform(0, 1, n)}
    micro = met.simulate_microstructure(comp, rng.uniform(720, 950, n), rng.uniform(1, 100, n))
    total = sum(micro[p] for p in ("f_ferrite", "f_bainite", "f_martensite", "f_retained_austenite"))
    np.testing.assert_allclose(total, 1.0, atol=1e-9)
    for p in ("f_ferrite", "f_bainite", "f_martensite", "f_retained_austenite"):
        assert (micro[p] >= -1e-12).all()


def test_faster_cooling_gives_more_martensite() -> None:
    comp = {"C": 0.15, "Mn": 1.8, "Si": 0.3}
    slow = met.simulate_microstructure(comp, 800.0, 2.0)
    fast = met.simulate_microstructure(comp, 800.0, 80.0)
    assert fast["f_martensite"] > slow["f_martensite"]


def test_equations_are_differentiable_in_torch() -> None:
    c = torch.tensor([0.1, 0.2], requires_grad=True)
    comp = {"C": c, "Mn": torch.tensor([1.5, 1.5])}
    phases = {p: torch.full((2,), 0.25) for p in
              ("f_ferrite", "f_bainite", "f_martensite", "f_retained_austenite")}
    _, uts = met.rule_of_mixtures(comp, phases, torch.tensor([5.0, 5.0]))
    uts.sum().backward()
    assert c.grad is not None and (c.grad > 0).all()
