from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pytest

from src.config import COMPOSITION_BOUNDS, ELEMENTS, INPUT_FEATURES, PHASES, PROCESS_BOUNDS, TARGETS
from src.data_pipeline.calphad_gen import (
    CalphadDataGenerator,
    EmpiricalThermoBackend,
    latin_hypercube,
    mole_to_wt_fraction,
    wt_to_mole_fraction,
)
from src.data_pipeline.preprocessing import (
    NormalizationStats,
    add_metallurgical_features,
    clean_dataset,
    compute_carbon_equivalent,
    remove_outliers,
)


def test_lhs_respects_bounds_and_stratifies() -> None:
    df = latin_hypercube(PROCESS_BOUNDS, 200, seed=0)
    for name, (lo, hi) in PROCESS_BOUNDS.items():
        assert df[name].between(lo, hi).all()
    # LHS: exatamente uma amostra por estrato em cada dimensão
    u = (df["T_anneal_C"] - 720) / (950 - 720)
    assert len(np.unique(np.floor(u * 200).clip(0, 199))) == 200


def test_lhs_fixed_variable() -> None:
    df = latin_hypercube({"C": (0.1, 0.2), "Ni": (0.0, 0.0)}, 10, seed=0)
    assert (df["Ni"] == 0.0).all()


def test_wt_mole_roundtrip() -> None:
    comp = {"C": 0.2, "Mn": 1.5, "Si": 0.5}
    back = mole_to_wt_fraction(wt_to_mole_fraction(comp))
    for el, w in comp.items():
        assert back[el] == pytest.approx(w, rel=1e-9)


def test_generator_schema_and_groups(synthetic_df: pd.DataFrame) -> None:
    for col in (*INPUT_FEATURES, *TARGETS, *PHASES, "alloy_id"):
        assert col in synthetic_df
    assert (synthetic_df["YS_MPa"] <= synthetic_df["UTS_MPa"]).all()
    np.testing.assert_allclose(synthetic_df[list(PHASES)].sum(axis=1), 1.0, atol=1e-6)
    # Mesma composição dentro de cada grupo
    assert (synthetic_df.groupby("alloy_id")[list(ELEMENTS)].nunique() == 1).all().all()
    for el, (lo, hi) in COMPOSITION_BOUNDS.items():
        assert synthetic_df[el].between(lo, hi).all()


def test_generator_is_reproducible() -> None:
    a = CalphadDataGenerator(seed=3).generate(20, 2)
    b = CalphadDataGenerator(seed=3).generate(20, 2)
    pd.testing.assert_frame_equal(a, b)


def test_empirical_backend_lever_rule() -> None:
    comp = pd.DataFrame([{el: 0.0 for el in ELEMENTS} | {"C": 0.2}] * 3)
    eq = EmpiricalThermoBackend().equilibrium_batch(comp, np.array([700.0, 800.0, 950.0]))
    f = eq["f_austenite_eq"].to_numpy()
    assert f[0] == 0.0 and 0.0 < f[1] < 1.0 and f[2] == 1.0


def test_clean_dataset_removes_invalid_rows(synthetic_df: pd.DataFrame) -> None:
    df = synthetic_df.head(20).copy()
    df.loc[0, "YS_MPa"] = df.loc[0, "UTS_MPa"] + 100  # violação física
    df.loc[1, "C"] = -0.1                               # composição negativa
    df.loc[2, "UTS_MPa"] = np.nan                       # alvo ausente
    df = df.drop(columns=["Cu"])                        # elemento não reportado
    out = clean_dataset(df)
    assert len(out) == 17
    assert (out["Cu"] == 0).all()


def test_isolation_forest_flags_gross_outlier(synthetic_df: pd.DataFrame) -> None:
    df = synthetic_df.copy()
    df.loc[0, "UTS_MPa"] = 1e5
    _, mask = remove_outliers(df, contamination=0.01)
    assert not mask[0]


def test_carbon_equivalent_column(synthetic_df: pd.DataFrame) -> None:
    ceq = compute_carbon_equivalent(synthetic_df)
    feats = add_metallurgical_features(synthetic_df)
    np.testing.assert_allclose(ceq, feats["Ceq_IIW"])
    assert (feats["Ac3_C"] > feats["Ac1_C"]).mean() > 0.95


def test_normalization_stats_log_feature(dataset) -> None:
    stats = NormalizationStats.fit(dataset.X, dataset.Y)
    assert "cooling_rate_Cps" in stats.log_features
    Xn = stats.transform_X(dataset.X)
    np.testing.assert_allclose(Xn.mean(axis=0), 0.0, atol=1e-4)


def _steel_test_tdb() -> str | None:
    hits = glob.glob("/usr/**/pycalphad/tests/databases/mc_fecocrnbti.tdb", recursive=True)
    return hits[0] if hits else None


@pytest.mark.slow
def test_pycalphad_backend_pure_iron_alpha_gamma() -> None:
    pytest.importorskip("pycalphad")
    tdb = _steel_test_tdb()
    if tdb is None:
        pytest.skip("Base TDB de teste do pycalphad não encontrada")
    from src.data_pipeline.calphad_gen import PyCalphadBackend

    backend = PyCalphadBackend(tdb)
    assert backend.equilibrium({}, 850.0).austenite_fraction == pytest.approx(0.0, abs=1e-6)
    assert backend.equilibrium({}, 1000.0).austenite_fraction == pytest.approx(1.0, abs=1e-6)
