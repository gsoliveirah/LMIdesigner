"""Dashboard interativo (Streamlit) do design inverso de ligas de Al secundárias.

Execução::

    streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import (  # noqa: E402
    COMPOSITION_BOUNDS,
    DEFAULT_MODEL_PATH,
    ELEMENTS,
    INPUT_FEATURES,
    LOG_FEATURES,
    NON_ADDABLE_ELEMENTS,
    PHASES,
    PROCESS_BOUNDS,
    PROCESS_FEATURES,
    SCRAP_GRADES,
    OptimizationConfig,
    TrainingConfig,
)
from src.data_pipeline.calphad_gen import (  # noqa: E402
    CalphadDataGenerator,
    EmpiricalThermoBackend,
    PyCalphadBackend,
    ThermoBackend,
)
from src.data_pipeline.preprocessing import preprocess  # noqa: E402
from src.models.pinn import PhysicsInformedNet, load_checkpoint, predict  # noqa: E402
from src.models.train import HyperParams, PINNTrainer  # noqa: E402
from src.optimization.charge import ChargeModel, optimize_charge_lp  # noqa: E402
from src.optimization.inverse_design import OBJECTIVES, InverseDesigner  # noqa: E402
from src.optimization.thermo_filter import ThermoFilter  # noqa: E402

UNITS = {"UTS_MPa": "UTS (MPa)", "YS_MPa": "YS (MPa)", "El_pct": "Alongamento (%)",
         "cost_USD_per_kg": "Custo (USD/kg)", "CO2_kg_per_kg": "CO₂ (kg/kg)",
         "scrap_fraction": "Fração de sucata"}
PHASE_LABELS = {"f_alpha_Al": "Al-α", "f_Si_eutectic": "Si eutético", "f_beta_AlFeSi": "β-Al5FeSi",
                "f_alpha_AlFeMnSi": "α-Al(Fe,Mn)Si", "f_Al2Cu": "Al2Cu"}

st.set_page_config(page_title="Recycled Al Inverse Design", page_icon="♻️", layout="wide")


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------
def get_model() -> PhysicsInformedNet | None:
    if "model" not in st.session_state and DEFAULT_MODEL_PATH.exists():
        st.session_state["model"], st.session_state["model_meta"] = load_checkpoint(DEFAULT_MODEL_PATH)
    return st.session_state.get("model")


def make_backend(tdb: str) -> ThermoBackend:
    return PyCalphadBackend(tdb) if tdb else EmpiricalThermoBackend()


def scrap_mix_widget(key: str) -> dict[str, float]:
    st.caption("Mistura de sucata (proporções relativas)")
    cols = st.columns(len(SCRAP_GRADES))
    mix = {g: cols[i].number_input(SCRAP_GRADES[g].name, 0.0, 1.0, 1.0 if g == "tense_cast" else 0.0,
                                   step=0.05, key=f"{key}_{g}")
           for i, g in enumerate(SCRAP_GRADES)}
    mix = {g: v for g, v in mix.items() if v > 0}
    return mix or {"tense_cast": 1.0}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("♻️ Recycled Al Inverse Design")
st.sidebar.caption("PINN + CALPHAD + NSGA-III · ligas Al-Si secundárias")
tdb_path = st.sidebar.text_input("Base TDB de Al (opcional, ativa pycalphad)", value="")
low_carbon = st.sidebar.toggle("Al primário de baixo carbono (hidro)", value=False)
model = get_model()
st.sidebar.markdown(f"**Modelo carregado:** {'✅' if model is not None else '❌ nenhum'}")
if model is not None and st.session_state.get("model_meta"):
    meta = st.session_state["model_meta"]
    st.sidebar.json({k: v for k, v in meta.items() if k in ("cv_score", "n_samples", "n_epochs")})

tab_data, tab_model, tab_pred, tab_inv, tab_charge = st.tabs(
    ["📊 Dados", "🧠 Modelo PINN", "🔬 Predição direta", "🎯 Design inverso", "♻️ Carga do forno"]
)

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
with tab_data:
    st.header("Geração / carga de dados")
    c1, c2, c3 = st.columns(3)
    n_alloys = c1.number_input("Nº de ligas (LHS)", 50, 20000, 800, step=50)
    n_proc = c2.number_input("Rotas de processo por liga", 1, 20, 4)
    seed = c3.number_input("Semente", 0, 10_000, 42)
    if st.button("Gerar dataset sintético", type="primary"):
        with st.spinner("Calculando equilíbrio e propriedades…"):
            gen = CalphadDataGenerator(backend=make_backend(tdb_path), seed=int(seed))
            st.session_state["raw"] = gen.generate(int(n_alloys), int(n_proc))
            gen.save(st.session_state["raw"])
    upload = st.file_uploader("…ou carregue dados experimentais (CSV)", type="csv")
    if upload is not None:
        st.session_state["raw"] = pd.read_csv(upload)

    if "raw" in st.session_state:
        raw = st.session_state["raw"]
        st.success(f"{len(raw)} amostras")
        st.dataframe(raw.head(200), width="stretch")
        if {"UTS_MPa", "El_pct", "Fe"} <= set(raw):
            fig = px.scatter(raw, x="El_pct", y="UTS_MPa", color="Fe", labels=UNITS | {"Fe": "Fe (wt%)"},
                             title="Resistência × ductilidade — efeito do Fe da sucata",
                             color_continuous_scale="Viridis", opacity=0.6, log_x=True)
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
with tab_model:
    st.header("Treinamento da PINN")
    c1, c2, c3, c4 = st.columns(4)
    n_trials = c1.number_input("Trials Optuna (0 = sem busca)", 0, 200, 0)
    epochs = c2.number_input("Épocas máx.", 10, 2000, 150)
    folds = c3.number_input("Folds (GroupKFold)", 2, 10, 3)
    lam = c4.number_input("λ física (sem Optuna)", 0.0, 10.0, 0.1, format="%.3f")
    if st.button("Treinar", type="primary", disabled="raw" not in st.session_state):
        with st.spinner("Treinando…"):
            ds = preprocess(st.session_state["raw"])
            st.info(f"Limpeza: {ds.report}")
            trainer = PINNTrainer(TrainingConfig(n_trials=int(n_trials), max_epochs=int(epochs),
                                                 n_splits=int(folds), patience=max(int(epochs) // 8, 5)))
            res = trainer.run(ds, tune=n_trials > 0, params=HyperParams(lambda_physics=lam),
                              save_path=DEFAULT_MODEL_PATH)
            st.session_state["model"] = res.model
            st.session_state["model_meta"] = {"cv_score": res.cv_score, "n_samples": len(ds.frame)}
            st.session_state["cv_metrics"] = res.cv_metrics
            st.session_state["val_data"] = ds
            model = res.model
    if "cv_metrics" in st.session_state:
        st.subheader("Validação cruzada agrupada por liga")
        st.dataframe(st.session_state["cv_metrics"].round(3), width="stretch")
    if model is not None and "val_data" in st.session_state:
        ds = st.session_state["val_data"]
        pred = predict(model, ds.X)
        target = st.selectbox("Alvo", model.target_names, index=1)
        fig = px.scatter(x=ds.Y[target], y=pred[target], opacity=0.4,
                         labels={"x": f"{target} observado", "y": f"{target} previsto"})
        lo, hi = float(ds.Y[target].min()), float(ds.Y[target].max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x"))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Predição direta
# ---------------------------------------------------------------------------
with tab_pred:
    st.header("Predição direta de uma liga")
    if model is None:
        st.warning("Treine ou carregue um modelo primeiro.")
    else:
        bounds = {**COMPOSITION_BOUNDS, **PROCESS_BOUNDS}
        # AlSi7Mg secundária típica (Fe/Cu/Zn/Ni/Cr são tetos herdados da sucata)
        defaults = {"Si": 7.0, "Fe": 0.3, "Cu": 0.5, "Mn": 0.15, "Mg": 0.35, "Zn": 0.4, "Cr": 0.03,
                    "Ni": 0.05, "Ti": 0.12, "Sr": 0.015, "cooling_rate_Cps": 2.0, "T_solution_C": 510.0,
                    "T_aging_C": 170.0, "t_aging_h": 6.0}
        values: dict[str, float] = {}
        st.subheader("Composição (wt%, Al = balanço)")
        cols = st.columns(5)
        for i, name in enumerate(ELEMENTS):
            lo, hi = bounds[name]
            values[name] = cols[i % 5].slider(name, float(lo), float(hi), float(defaults.get(name, lo)),
                                              step=float((hi - lo) / 200), format="%.3f")
        st.subheader("Processo")
        cols = st.columns(4)
        for i, name in enumerate(PROCESS_FEATURES):
            lo, hi = bounds[name]
            default = defaults.get(name, (lo * hi) ** 0.5 if name in LOG_FEATURES else (lo + hi) / 2)
            values[name] = cols[i].slider(name, float(lo), float(hi), float(default),
                                          step=float((hi - lo) / 200), format="%.2f")
        pred_mix = scrap_mix_widget("pred")
        cand = pd.DataFrame([values])[list(INPUT_FEATURES)]
        charge = ChargeModel.from_mix(pred_mix, low_carbon_primary=low_carbon).compute(cand).iloc[0]
        for el in NON_ADDABLE_ELEMENTS:  # teor real entregue pela carga
            cand[el] = charge[f"achieved_{el}"]
        pred = predict(model, cand, n_mc_samples=30).iloc[0]
        m = st.columns(6)
        m[0].metric("YS (MPa)", f"{pred['YS_MPa']:.0f}", f"± {pred['YS_MPa_std']:.0f}")
        m[1].metric("UTS (MPa)", f"{pred['UTS_MPa']:.0f}", f"± {pred['UTS_MPa_std']:.0f}")
        m[2].metric("Alongamento (%)", f"{pred['El_pct']:.1f}", f"± {pred['El_pct_std']:.1f}")
        m[3].metric("Sucata na carga (máx.)", f"{100 * charge['scrap_fraction']:.0f} %")
        m[4].metric("Custo (USD/kg)", f"{charge['cost_USD_per_kg']:.2f}")
        m[5].metric("CO₂ (kg/kg)", f"{charge['CO2_kg_per_kg']:.2f}")

        c1, c2 = st.columns(2)
        fig = px.bar(x=[PHASE_LABELS[p] for p in PHASES[1:]], y=[100 * pred[p] for p in PHASES[1:]],
                     labels={"x": "Fase", "y": "Fração (%)"},
                     title=f"Microestrutura prevista (Al-α = {100 * pred['f_alpha_Al']:.1f} %)")
        c1.plotly_chart(fig, width="stretch")
        result = ThermoFilter().evaluate(cand.iloc[0].to_dict() | {p: float(pred[p]) for p in PHASES})
        c2.subheader("Filtro termodinâmico / risco")
        c2.json({k: round(v, 3) for k, v in result.metrics.items()})
        c2.markdown(f"**Aprovado:** {'✅' if result.passed else '❌'} · **Risk score:** {result.risk_score:.2f}")
        for f in result.flags:
            show = {"CRITICAL": c2.error, "WARNING": c2.warning}.get(f.severity.name, c2.info)
            show(f"**{f.code}** — {f.message} (valor={f.value:.3g}, limite={f.limit:.3g})")

# ---------------------------------------------------------------------------
# Design inverso
# ---------------------------------------------------------------------------
with tab_inv:
    st.header("Design inverso multiobjetivo (NSGA-III)")
    if model is None:
        st.warning("Treine ou carregue um modelo primeiro.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        min_uts = c1.number_input("UTS mínimo (MPa)", 100.0, 500.0, 260.0, step=5.0)
        min_el = c2.number_input("Alongamento mínimo (%)", 0.0, 20.0, 4.0)
        min_ys = c3.number_input("YS mínimo (MPa)", 0.0, 400.0, 0.0, step=5.0)
        min_rec = c4.slider("Fração mínima de sucata", 0.0, 1.0, 0.0, 0.05)
        mix = scrap_mix_widget("inv")
        c1, c2, c3, c4 = st.columns(4)
        fix_cr = c1.toggle("Fixar taxa de resfriamento (processo da fundição)", value=False)
        cr_value = c2.number_input("Taxa de resfriamento (°C/s)", 0.1, 50.0, 2.0, disabled=not fix_cr)
        pop = c3.number_input("População", 40, 1000, 120)
        gens = c4.number_input("Gerações", 5, 2000, 100)
        mc = st.number_input("Amostras MC-dropout (0 = off)", 0, 50, 0)
        st.caption("Pesos para a solução de compromisso (ASF)")
        w = st.columns(4)
        weights = [w[i].slider(UNITS[o], 0.0, 1.0, 0.25) for i, o in enumerate(OBJECTIVES)]

        if st.button("Otimizar", type="primary"):
            cfg = OptimizationConfig(
                pop_size=int(pop), n_generations=int(gens), min_uts_MPa=min_uts, min_elongation_pct=min_el,
                min_ys_MPa=min_ys, min_recycled_fraction=min_rec, scrap_mix=mix,
                low_carbon_primary=low_carbon, n_mc_samples=int(mc),
                fixed_values={"cooling_rate_Cps": cr_value} if fix_cr else {},
            )
            with st.spinner("Executando NSGA-III…"):
                st.session_state["design"] = InverseDesigner(model, cfg).run()

        if "design" in st.session_state:
            res = st.session_state["design"]
            pareto = res.pareto
            if not res.feasible:
                st.error("Nenhuma solução viável — exibindo as de menor violação. Relaxe as restrições.")
            st.success(f"{len(pareto)} soluções · {res.n_evaluations} avaliações")
            best = InverseDesigner.select_compromise(pareto, np.maximum(weights, 1e-3))
            fig = px.scatter(pareto, x="El_pct", y="UTS_MPa", color="CO2_kg_per_kg",
                             size="cost_USD_per_kg",
                             hover_data=list(ELEMENTS) + list(PROCESS_FEATURES) + ["scrap_fraction"],
                             labels=UNITS, title="Fronteira de Pareto (cor = CO₂, tamanho = custo)",
                             color_continuous_scale="Viridis")
            fig.add_trace(go.Scatter(x=[best["El_pct"]], y=[best["UTS_MPa"]], mode="markers",
                                     marker=dict(symbol="star", size=20, color="black"), name="Compromisso"))
            st.plotly_chart(fig, width="stretch")
            st.plotly_chart(px.scatter(pareto, x="scrap_fraction", y="CO2_kg_per_kg", color="Fe",
                                       labels=UNITS, title="Conteúdo reciclado × CO₂"), width="stretch")
            dims = ["Si", "Fe", "Cu", "Mn", "Mg", "T_solution_C", "UTS_MPa", "El_pct",
                    "scrap_fraction", "CO2_kg_per_kg"]
            st.plotly_chart(px.parallel_coordinates(pareto, dimensions=dims, color="UTS_MPa",
                                                    color_continuous_scale="Viridis"), width="stretch")
            st.subheader("⭐ Solução de compromisso")
            st.dataframe(best.to_frame().T, width="stretch")
            st.subheader("Conjunto de Pareto")
            st.dataframe(pareto, width="stretch")
            st.download_button("Baixar CSV", pareto.to_csv(index=False), "pareto_front.csv", "text/csv")
            if tdb_path and st.button("Validar Pareto com CALPHAD"):
                with st.spinner("Equilíbrio CALPHAD nas soluções…"):
                    val = ThermoFilter().validate_with_calphad(pareto, make_backend(tdb_path))
                st.dataframe(val[["f_liquid_eq", "f_sludge_eq", "risk_score", "passed", "flags"]])

# ---------------------------------------------------------------------------
# Carga do forno (LP)
# ---------------------------------------------------------------------------
with tab_charge:
    st.header("Carga ótima multi-sucata (programação linear)")
    st.caption("Fe, Cu, Zn e Ni da sucata não podem ser removidos — só diluídos com Al primário.")
    source = st.radio("Composição-alvo", ["Solução de compromisso do design inverso", "Manual"],
                      horizontal=True)
    if source.startswith("Solução") and "design" in st.session_state:
        best = InverseDesigner.select_compromise(st.session_state["design"].pareto)
        target = {el: float(best[el]) for el in ELEMENTS}
    else:
        cols = st.columns(5)
        base = {"Si": 7.0, "Fe": 0.4, "Cu": 0.5, "Mn": 0.2, "Mg": 0.35, "Zn": 0.3, "Ti": 0.1, "Sr": 0.015}
        target = {el: cols[i % 5].number_input(f"{el} (wt%)", 0.0, 15.0, base.get(el, 0.0), step=0.01,
                                               key=f"lp_{el}") for i, el in enumerate(ELEMENTS)}
    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox("Objetivo", ["cost", "co2", "weighted"])
    co2_price = c2.number_input("Preço do carbono (USD/kg CO₂)", 0.0, 1.0, 0.1, disabled=objective != "weighted")
    tol = c3.number_input("Tolerância ± (wt%) para Fe/Cu/Zn/Mn", 0.0, 1.0, 0.05, step=0.01)
    res = optimize_charge_lp(target, objective=objective, co2_weight=co2_price, low_carbon_primary=low_carbon,
                             tolerance_wt={el: tol for el in ("Fe", "Cu", "Zn", "Mn")})
    if not res.success:
        st.error(f"Carga inviável: {res.message}")
    else:
        m = st.columns(3)
        m[0].metric("Conteúdo reciclado", f"{100 * res.scrap_fraction:.1f} %")
        m[1].metric("Custo (USD/kg)", f"{res.cost_USD_per_kg:.3f}")
        m[2].metric("CO₂ (kg/kg)", f"{res.CO2_kg_per_kg:.2f}")
        frac = pd.Series(res.fractions)
        frac = frac[frac > 1e-5].sort_values(ascending=False) * 1000
        st.plotly_chart(px.bar(x=frac.index, y=frac.values, labels={"x": "Insumo", "y": "kg por t de banho"}),
                        width="stretch")
