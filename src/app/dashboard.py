"""Dashboard interativo (Streamlit) do design inverso de AHSS.

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
    ELEMENTS,
    INPUT_FEATURES,
    LOG_FEATURES,
    MODELS_DIR,
    PHASES,
    PROCESS_BOUNDS,
    OptimizationConfig,
    TrainingConfig,
)
from src.data_pipeline.calphad_gen import (  # noqa: E402
    CalphadDataGenerator,
    EmpiricalThermoBackend,
    PyCalphadBackend,
)
from src.data_pipeline.preprocessing import preprocess  # noqa: E402
from src.models.pinn import PhysicsInformedNet, load_checkpoint, predict  # noqa: E402
from src.models.train import HyperParams, PINNTrainer  # noqa: E402
from src.optimization.inverse_design import (  # noqa: E402
    OBJECTIVES,
    InverseDesigner,
    alloy_co2,
    alloy_cost,
)
from src.optimization.thermo_filter import ThermoFilter  # noqa: E402

DEFAULT_MODEL = MODELS_DIR / "pinn_ahss.pt"
UNITS = {"UTS_MPa": "UTS (MPa)", "YS_MPa": "YS (MPa)", "El_pct": "Alongamento (%)",
         "cost_USD_per_kg": "Custo (USD/kg)", "CO2_kg_per_kg": "CO₂ (kg/kg)"}

st.set_page_config(page_title="AHSS Inverse Design", page_icon="⚙️", layout="wide")


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------
def get_model() -> PhysicsInformedNet | None:
    if "model" not in st.session_state and DEFAULT_MODEL.exists():
        st.session_state["model"], st.session_state["model_meta"] = load_checkpoint(DEFAULT_MODEL)
    return st.session_state.get("model")


def make_backend(tdb_path: str):
    return PyCalphadBackend(tdb_path) if tdb_path else EmpiricalThermoBackend()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ AHSS Inverse Design")
st.sidebar.caption("PINN + CALPHAD + NSGA-III")
tdb_path = st.sidebar.text_input("Base TDB (opcional, ativa pycalphad)", value="")
model = get_model()
st.sidebar.markdown(f"**Modelo carregado:** {'✅' if model is not None else '❌ nenhum'}")
if model is not None and st.session_state.get("model_meta"):
    meta = st.session_state["model_meta"]
    st.sidebar.json({k: v for k, v in meta.items() if k in ("cv_score", "n_samples", "n_epochs")})

tab_data, tab_model, tab_pred, tab_inv = st.tabs(
    ["📊 Dados", "🧠 Modelo PINN", "🔬 Predição direta", "🎯 Design inverso"]
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
        if {"UTS_MPa", "El_pct"} <= set(raw):
            fig = px.scatter(raw, x="UTS_MPa", y="El_pct",
                             color="f_martensite" if "f_martensite" in raw else None,
                             labels=UNITS, title="Banana curve: resistência × ductilidade",
                             color_continuous_scale="Viridis", opacity=0.6)
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
                              save_path=DEFAULT_MODEL)
            st.session_state["model"] = res.model
            st.session_state["model_meta"] = {"cv_score": res.cv_score, "n_samples": len(ds.frame)}
            st.session_state["cv_metrics"] = res.cv_metrics
            st.session_state["val_data"] = ds
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
        cols = st.columns(4)
        values: dict[str, float] = {}
        bounds = {**COMPOSITION_BOUNDS, **PROCESS_BOUNDS}
        for i, name in enumerate(INPUT_FEATURES):
            lo, hi = bounds[name]
            default = (lo * hi) ** 0.5 if name in LOG_FEATURES else (lo + hi) / 2
            step = (hi - lo) / 200 if hi > lo else 0.001
            values[name] = cols[i % 4].slider(name, float(lo), float(hi), float(default),
                                              step=float(step), format="%.4f" if hi < 0.1 else "%.3f")
        cand = pd.DataFrame([values])
        pred = predict(model, cand, n_mc_samples=30).iloc[0]
        m = st.columns(5)
        m[0].metric("YS (MPa)", f"{pred['YS_MPa']:.0f}", f"± {pred['YS_MPa_std']:.0f}")
        m[1].metric("UTS (MPa)", f"{pred['UTS_MPa']:.0f}", f"± {pred['UTS_MPa_std']:.0f}")
        m[2].metric("Alongamento (%)", f"{pred['El_pct']:.1f}", f"± {pred['El_pct_std']:.1f}")
        m[3].metric("Custo (USD/kg)", f"{alloy_cost(cand)[0]:.3f}")
        m[4].metric("CO₂ (kg/kg)", f"{alloy_co2(cand)[0]:.2f}")

        c1, c2 = st.columns(2)
        fig = px.pie(names=[p.replace("f_", "") for p in PHASES], values=[pred[p] for p in PHASES],
                     title="Microestrutura prevista", hole=0.4)
        c1.plotly_chart(fig, width="stretch")
        result = ThermoFilter(backend=make_backend(tdb_path)).evaluate(values)
        c2.subheader("Filtro termodinâmico / risco")
        c2.json({k: round(v, 3) for k, v in result.metrics.items()})
        c2.markdown(f"**Aprovado:** {'✅' if result.passed else '❌'} · **Risk score:** {result.risk_score:.2f}")
        for f in result.flags:
            (c2.error if f.severity.name == "CRITICAL" else c2.warning)(
                f"**{f.code}** — {f.message} (valor={f.value:.3g}, limite={f.limit:.3g})")

# ---------------------------------------------------------------------------
# Design inverso
# ---------------------------------------------------------------------------
with tab_inv:
    st.header("Design inverso multiobjetivo (NSGA-III)")
    if model is None:
        st.warning("Treine ou carregue um modelo primeiro.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        min_uts = c1.number_input("UTS mínimo (MPa)", 300.0, 2500.0, 980.0, step=10.0)
        min_el = c2.number_input("Alongamento mínimo (%)", 0.0, 50.0, 12.0)
        ceq_max = c3.number_input("C_eq IIW máx.", 0.1, 1.5, 0.60)
        fe_route = c4.selectbox("Rota do Fe (CO₂)", ["BF-BOF", "EAF"])
        c1, c2, c3 = st.columns(3)
        pop = c1.number_input("População", 40, 1000, 120)
        gens = c2.number_input("Gerações", 5, 2000, 100)
        mc = c3.number_input("Amostras MC-dropout (0 = off)", 0, 50, 0)
        st.caption("Pesos para a solução de compromisso (ASF)")
        w = st.columns(4)
        weights = [w[i].slider(UNITS[o], 0.0, 1.0, 0.25) for i, o in enumerate(OBJECTIVES)]

        if st.button("Otimizar", type="primary"):
            cfg = OptimizationConfig(pop_size=int(pop), n_generations=int(gens), min_uts_MPa=min_uts,
                                     min_elongation_pct=min_el, ceq_max=ceq_max, fe_route=fe_route,
                                     n_mc_samples=int(mc))
            with st.spinner("Executando NSGA-III…"):
                st.session_state["design"] = InverseDesigner(model, cfg).run()

        if "design" in st.session_state:
            res = st.session_state["design"]
            pareto = res.pareto
            if not res.feasible:
                st.error("Nenhuma solução viável — exibindo as de menor violação. Relaxe as restrições.")
            st.success(f"{len(pareto)} soluções · {res.n_evaluations} avaliações")
            best = InverseDesigner.select_compromise(pareto, np.maximum(weights, 1e-3))
            fig = px.scatter(pareto, x="UTS_MPa", y="El_pct", color="cost_USD_per_kg",
                             size="CO2_kg_per_kg", hover_data=list(ELEMENTS) + ["T_anneal_C", "cooling_rate_Cps"],
                             labels=UNITS, title="Fronteira de Pareto", color_continuous_scale="Plasma")
            fig.add_trace(go.Scatter(x=[best["UTS_MPa"]], y=[best["El_pct"]], mode="markers",
                                     marker=dict(symbol="star", size=20, color="black"), name="Compromisso"))
            st.plotly_chart(fig, width="stretch")
            dims = ["C", "Mn", "Si", "Al", "T_anneal_C", "UTS_MPa", "El_pct", "cost_USD_per_kg", "CO2_kg_per_kg"]
            st.plotly_chart(px.parallel_coordinates(pareto, dimensions=dims, color="UTS_MPa",
                                                    color_continuous_scale="Viridis"),
                            width="stretch")
            st.subheader("⭐ Solução de compromisso")
            st.dataframe(best.to_frame().T, width="stretch")
            st.subheader("Conjunto de Pareto")
            st.dataframe(pareto, width="stretch")
            st.download_button("Baixar CSV", pareto.to_csv(index=False), "pareto_front.csv", "text/csv")
            if tdb_path and st.button("Validar Pareto com CALPHAD"):
                with st.spinner("Equilíbrio CALPHAD nas soluções…"):
                    val = ThermoFilter().validate_with_calphad(pareto, make_backend(tdb_path))
                st.dataframe(val[["f_austenite_eq", "f_undesired_eq", "risk_score", "passed", "flags"]])
