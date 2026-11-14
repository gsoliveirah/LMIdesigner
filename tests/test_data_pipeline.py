from __future__ import annotations

import glob

import numpy as np
import pandas as pd
import pytest

from src.config import COMPOSITION_BOUNDS, ELEMENTS, INPUT_FEATURES, PHASES, PROCESS_BOUNDS, TARGETS
from src import metallurgy as met
from src.data_pipeline.calphad_gen import (
    EQ_COLUMNS,
    CalphadDataGenerator,
    EmpiricalThermoBackend,
    classify_phase,
    latin_hypercube,
    mole_to_wt_fraction,
    wt_to_mole_fraction,
)
from src.data_pipeline.preprocessing import (
    NormalizationStats,
    add_metallurgical_features,
    clean_dataset,
    compute_c_eq,
    remove_outliers,
)


def test_lhs_respects_bounds_and_stratifies() -> None:
    df = latin_hypercube(PROCESS_BOUNDS, 200, seed=0)
    for name, (lo, hi) in PROCESS_BOUNDS.items():
        assert df[name].between(lo, hi).all()
    # LHS: exatamente uma amostra por estrato em cada dimensão
    u = (df["T_solution_C"] - 480) / (550 - 480)
    assert len(np.unique(np.floor(u * 200).clip(0, 199))) == 200


def test_lhs_fixed_variable() -> None:
    df = latin_hypercube({"Si": (6.0, 8.0), "Ni": (0.0, 0.0)}, 10, seed=0)
    assert (df["Ni"] == 0.0).all()


def test_wt_mole_roundtrip() -> None:
    comp = {"Si": 7.0, "Fe": 0.5, "Cu": 3.0, "Mg": 0.3}
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
    for name, (lo, hi) in PROCESS_BOUNDS.items():
        assert synthetic_df[name].between(lo, hi).all()
    assert (synthetic_df[list(PHASES)] >= 0).all().all()


def test_generator_is_reproducible() -> None:
    a = CalphadDataGenerator(seed=3).generate(20, 2)
    b = CalphadDataGenerator(seed=3).generate(20, 2)
    pd.testing.assert_frame_equal(a, b)


def test_solution_temperature_follows_incipient_melting(synthetic_df: pd.DataFrame) -> None:
    """A T de solubilização é amostrada relativa ao eutético: maioria segura, alguns queimados."""
    t_inc = np.asarray(met.incipient_melting_temperature({el: synthetic_df[el] for el in ELEMENTS}))
    burnt = (synthetic_df["T_solution_C"] > t_inc).mean()
    assert 0.02 < burnt < 0.4


def test_empirical_backend_dissolution_and_sludge() -> None:
    comp = pd.DataFrame([
        {el: 0.0 for el in ELEMENTS} | {"Si": 7.0, "Cu": 3.5, "Mg": 0.3, "Fe": 0.3},  # 319: Cu dissolve
        {el: 0.0 for el in ELEMENTS} | {"Si": 7.0, "Cu": 3.5, "Mg": 0.3, "Fe": 0.3},  # queimada a 540 °C
        {el: 0.0 for el in ELEMENTS} | {"Si": 9.0, "Fe": 1.2, "Mn": 0.6, "Cr": 0.15},  # SF = 2.85: sludge
    ])
    eq = EmpiricalThermoBackend().equilibrium_batch(comp, np.array([495.0, 540.0, 540.0]))
    assert list(eq.columns) == list(EQ_COLUMNS)
    assert eq.loc[0, "Cu_fcc_eq_wt"] == pytest.approx(3.5)
    assert eq.loc[0, "f_liquid_eq"] == 0.0 and eq.loc[1, "f_liquid_eq"] > 0.0
    assert eq.loc[2, "f_sludge_eq"] > 0.0 and eq.loc[0, "f_sludge_eq"] == 0.0


def test_empirical_single_point_matches_batch() -> None:
    comp = {"Si": 9.0, "Fe": 1.2, "Mn": 0.6, "Cr": 0.15, "Cu": 2.0, "Mg": 0.3}
    backend = EmpiricalThermoBackend()
    single = backend.solution_and_holding(comp, 500.0)
    batch = backend.equilibrium_batch(pd.DataFrame([{el: comp.get(el, 0.0) for el in ELEMENTS}]),
                                      np.array([500.0])).iloc[0]
    for col in EQ_COLUMNS:
        assert single[col] == pytest.approx(batch[col])


def test_classify_calphad_phase_names() -> None:
    assert classify_phase("ALCU_THETA") == "Al2Cu"
    assert classify_phase("AL5FESI") == "beta_AlFeSi"
    assert classify_phase("AL15SI2M4") == "alpha_AlFeMnSi"
    assert classify_phase("DIAMOND_A4") == "Si"
    assert classify_phase("FCC_A1#2") == "fcc"
    assert classify_phase("LIQUID") == "liquid"
    assert classify_phase("AL13FE4") == "Fe_other"


def test_clean_dataset_removes_invalid_rows(synthetic_df: pd.DataFrame) -> None:
    df = synthetic_df.head(20).copy()
    df.loc[0, "YS_MPa"] = df.loc[0, "UTS_MPa"] + 100  # violação física
    df.loc[1, "Si"] = -0.1                              # composição negativa
    df.loc[2, "UTS_MPa"] = np.nan                       # alvo ausente
    df.loc[3, "t_aging_h"] = 0.0                        # tempo inválido
    df = df.drop(columns=["Sr"])                        # elemento não reportado
    out = clean_dataset(df)
    assert len(out) == 16
    assert (out["Sr"] == 0).all()


def test_isolation_forest_flags_gross_outlier(synthetic_df: pd.DataFrame) -> None:
    df = synthetic_df.copy()
    df.loc[0, "UTS_MPa"] = 1e5
    _, mask = remove_outliers(df, contamination=0.01)
    assert not mask[0]  # erro grosseiro univariado (gate MAD)


def test_isolation_forest_flags_multivariate_outlier(synthetic_df: pd.DataFrame) -> None:
    df = synthetic_df.copy()
    df.loc[0, ["YS_MPa", "UTS_MPa", "El_pct"]] = [450.0, 520.0, 25.0]  # plausível isoladamente, não em conjunto
    _, mask = remove_outliers(df, contamination=0.01, gross_error_threshold=None)
    assert not mask[0]


def test_c_eq_methods(synthetic_df: pd.DataFrame) -> None:
    si_eq = compute_c_eq(synthetic_df)  # padrão: Si equivalente
    feats = add_metallurgical_features(synthetic_df)
    np.testing.assert_allclose(si_eq, feats["Si_eq"])
    assert (feats["Si_eq"] >= feats["Si"]).all()
    np.testing.assert_allclose(compute_c_eq(synthetic_df, "sludge"),
                               synthetic_df["Fe"] + 2 * synthetic_df["Mn"] + 3 * synthetic_df["Cr"])
    assert (feats["T_liquidus_C"] > feats["T_incipient_C"]).all()
    for col in ("Mn_Fe_ratio", "Fe_excess", "SDAS_um", "solution_margin_C", "log10_aging_P"):
        assert col in feats and feats[col].notna().all()
    with pytest.raises(ValueError):
        compute_c_eq(synthetic_df, "IIW")


def test_normalization_stats_log_feature(dataset) -> None:
    stats = NormalizationStats.fit(dataset.X, dataset.Y)
    assert set(stats.log_features) == {"cooling_rate_Cps", "t_aging_h"}
    Xn = stats.transform_X(dataset.X)
    np.testing.assert_allclose(Xn.mean(axis=0), 0.0, atol=1e-4)


def _pycalphad_test_tdb(name: str) -> str | None:
    hits = glob.glob(f"/usr/**/pycalphad/tests/databases/{name}", recursive=True)
    return hits[0] if hits else None


@pytest.mark.slow
def test_pycalphad_backend_al_cu_solvus() -> None:
    """Al-4Cu: θ-Al2Cu a 450 °C, solução sólida monofásica a 540 °C, líquido a 700 °C."""
    pytest.importorskip("pycalphad")
    tdb = _pycalphad_test_tdb("Al-Cu-Y.tdb")
    if tdb is None:
        pytest.skip("Base TDB de teste do pycalphad não encontrada")
    from src.data_pipeline.calphad_gen import PyCalphadBackend

    backend = PyCalphadBackend(tdb, phases=["LIQUID", "FCC_A1", "ALCU_THETA"])
    low = backend.equilibrium({"Cu": 4.0}, 450.0)
    assert low.fraction("Al2Cu") > 0.02
    # solvus real ~2.4 % Cu a 450 °C; o solvus empírico deve concordar em ±0.5 wt%
    assert 1.5 < low.matrix_composition_wt["Cu"] < 3.0
    assert float(met.cu_solubility(450.0)) == pytest.approx(low.matrix_composition_wt["Cu"], abs=0.5)
    high = backend.equilibrium({"Cu": 4.0}, 540.0)
    assert high.fraction("Al2Cu") == pytest.approx(0.0, abs=1e-6)
    assert high.matrix_composition_wt["Cu"] == pytest.approx(4.0, abs=0.05)
    assert backend.equilibrium({"Cu": 4.0}, 700.0).fraction("liquid") == pytest.approx(1.0, abs=1e-6)
    row = backend.solution_and_holding({"Cu": 4.0, "Si": 7.0}, 540.0)  # Si ausente da base: ignorado
    assert row["eq_converged"] == 1.0 and row["Cu_fcc_eq_wt"] == pytest.approx(4.0, abs=0.1)
