"""Geração de dados sintéticos via CALPHAD + Latin Hypercube Sampling.

Fluxo:
    LHS (composição × processo) -> equilíbrio termodinâmico na temperatura de
    recozimento (pycalphad ou backend empírico) -> modelo de transformação no
    resfriamento -> regra das misturas -> propriedades mecânicas + ruído.

Dois backends termodinâmicos estão disponíveis:

* :class:`PyCalphadBackend` — equilíbrio Gibbs completo com uma base TDB
  (ex.: TCFE, ``mc_fe`` do MatCalc). Lento (~s/ponto): use para datasets de
  validação ou centenas de pontos.
* :class:`EmpiricalThermoBackend` — aproximação vetorizada (Andrews +
  regra da alavanca). Instantânea; útil para prototipagem e testes, mas NÃO
  substitui CALPHAD para conclusões quantitativas.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src import metallurgy as met
from src.config import (
    ATOMIC_MASS,
    COMPOSITION_BOUNDS,
    ELEMENTS,
    KELVIN_OFFSET,
    LOG_FEATURES,
    PHASES,
    PRESSURE_PA,
    PROCESS_BOUNDS,
    PROCESS_FEATURES,
    SYNTHETIC_DIR,
    TARGETS,
)

logger = logging.getLogger(__name__)

# Fases consideradas indesejáveis em AHSS (fragilização / perda de ductilidade)
UNDESIRED_PHASES: tuple[str, ...] = ("SIGMA", "LAVES_PHASE", "LAVES", "MU_PHASE", "CHI_A12", "G_PHASE")
DEFAULT_STEEL_PHASES: tuple[str, ...] = (
    "LIQUID", "FCC_A1", "BCC_A2", "CEMENTITE", "M7C3", "M23C6", "M6C",
    "HCP_A3", "SIGMA", "LAVES_PHASE", "MU_PHASE", "AL4C3", "MNS", "TIN",
)


# ---------------------------------------------------------------------------
# Conversões
# ---------------------------------------------------------------------------
def wt_to_mole_fraction(composition: Mapping[str, float]) -> dict[str, float]:
    """Converte wt% (Fe = balanço) para fração molar."""
    wt = {el: float(composition.get(el, 0.0)) for el in ELEMENTS}
    wt["Fe"] = 100.0 - sum(wt.values())
    if wt["Fe"] <= 0:
        raise ValueError("Soma dos elementos de liga excede 100 wt%.")
    moles = {el: w / ATOMIC_MASS[el] for el, w in wt.items() if w > 0}
    total = sum(moles.values())
    return {el: n / total for el, n in moles.items()}


def mole_to_wt_fraction(x: Mapping[str, float]) -> dict[str, float]:
    """Converte fração molar para wt%."""
    masses = {el: xi * ATOMIC_MASS[el] for el, xi in x.items() if el in ATOMIC_MASS}
    total = sum(masses.values())
    return {el: 100.0 * m / total for el, m in masses.items()}


# ---------------------------------------------------------------------------
# Backends termodinâmicos
# ---------------------------------------------------------------------------
@dataclass
class EquilibriumResult:
    """Resultado de um cálculo de equilíbrio em (composição, T)."""

    phase_fractions: dict[str, float]
    austenite_fraction: float
    austenite_carbon_wt: float | None = None
    undesired_fraction: float = 0.0
    converged: bool = True
    extra: dict[str, float] = field(default_factory=dict)


class ThermoBackend(ABC):
    """Interface de um backend termodinâmico."""

    name: str = "abstract"

    @abstractmethod
    def equilibrium(self, composition: Mapping[str, float], T_C: float) -> EquilibriumResult:
        """Equilíbrio para uma liga em uma temperatura (°C)."""

    def equilibrium_batch(self, compositions: pd.DataFrame, T_C: np.ndarray) -> pd.DataFrame:
        """Equilíbrio para várias ligas. Implementação padrão: laço."""
        rows: list[dict[str, float]] = []
        for (_, comp), t in zip(compositions.iterrows(), np.asarray(T_C, dtype=float)):
            try:
                res = self.equilibrium(comp.to_dict(), float(t))
                rows.append({
                    "f_austenite_eq": res.austenite_fraction,
                    "C_austenite_eq_wt": np.nan if res.austenite_carbon_wt is None else res.austenite_carbon_wt,
                    "f_undesired_eq": res.undesired_fraction,
                    "eq_converged": float(res.converged),
                })
            except Exception as exc:  # noqa: BLE001 — falhas de convergência
                logger.warning("Equilíbrio falhou (%s): %s", self.name, exc)
                rows.append({"f_austenite_eq": np.nan, "C_austenite_eq_wt": np.nan,
                             "f_undesired_eq": np.nan, "eq_converged": 0.0})
        return pd.DataFrame(rows, index=compositions.index)


class EmpiricalThermoBackend(ThermoBackend):
    """Aproximação empírica vetorizada do equilíbrio α/γ/cementita.

    f_γ = (T − Ac1)/(Ac3 − Ac1) (alavanca linear), C_γ por balanço de massa.
    """

    name = "empirical"

    def equilibrium(self, composition: Mapping[str, float], T_C: float) -> EquilibriumResult:
        df = pd.DataFrame([dict(composition)])
        out = self.equilibrium_batch(df, np.array([T_C])).iloc[0]
        f_g = float(out["f_austenite_eq"])
        f_cem = float(np.clip((1 - f_g) * composition.get("C", 0.0) / 6.67, 0, 1))
        return EquilibriumResult(
            phase_fractions={"FCC_A1": f_g, "BCC_A2": 1 - f_g - f_cem, "CEMENTITE": f_cem},
            austenite_fraction=f_g,
            austenite_carbon_wt=float(out["C_austenite_eq_wt"]),
        )

    def equilibrium_batch(self, compositions: pd.DataFrame, T_C: np.ndarray) -> pd.DataFrame:
        comp = {el: compositions[el].to_numpy(float) for el in ELEMENTS if el in compositions}
        ac1 = np.asarray(met.ac1_andrews(comp), dtype=float)
        ac3 = np.asarray(met.ac3_andrews(comp), dtype=float)
        T = np.asarray(T_C, dtype=float)
        f_g = np.clip((T - ac1) / np.clip(ac3 - ac1, 1.0, None), 0.0, 1.0)
        c_g = np.clip(comp["C"] / np.clip(f_g, 0.05, 1.0), 0.0, 1.2)
        return pd.DataFrame(
            {"f_austenite_eq": f_g, "C_austenite_eq_wt": c_g,
             "f_undesired_eq": np.zeros_like(f_g), "eq_converged": np.ones_like(f_g)},
            index=compositions.index,
        )


class PyCalphadBackend(ThermoBackend):
    """Equilíbrio via pycalphad com uma base termodinâmica TDB.

    Parameters
    ----------
    tdb_path:
        Caminho para a base TDB (ex.: ``mc_fe_v2.060.tdb`` do MatCalc).
    phases:
        Fases a considerar. Padrão: fases típicas de aços presentes na base.
    min_mole_fraction:
        Elementos abaixo deste valor são descartados (estabilidade numérica).
    """

    name = "pycalphad"

    def __init__(
        self,
        tdb_path: str | Path,
        phases: Sequence[str] | None = None,
        min_mole_fraction: float = 1e-5,
    ) -> None:
        try:
            from pycalphad import Database
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pycalphad não instalado: pip install pycalphad") from exc
        self.db = Database(str(tdb_path))
        self.min_mole_fraction = min_mole_fraction
        db_phases = set(self.db.phases.keys())
        wanted = phases if phases is not None else DEFAULT_STEEL_PHASES
        self.phases = sorted(p for p in wanted if p in db_phases)
        if not self.phases:
            raise ValueError("Nenhuma das fases solicitadas existe na base TDB.")

    def equilibrium(self, composition: Mapping[str, float], T_C: float) -> EquilibriumResult:
        from pycalphad import equilibrium, variables as v

        x = wt_to_mole_fraction(composition)
        db_elements = {str(e).upper() for e in self.db.elements}
        solutes = [
            el for el, xi in x.items()
            if el != "Fe" and xi >= self.min_mole_fraction and el.upper() in db_elements
        ]
        comps = ["FE", "VA"] + [el.upper() for el in solutes]
        conds: dict = {v.T: T_C + KELVIN_OFFSET, v.P: PRESSURE_PA, v.N: 1}
        for el in solutes:
            conds[v.X(el.upper())] = x[el]

        eq = equilibrium(self.db, comps, self.phases, conds)
        phase_names = np.atleast_1d(eq.Phase.values.squeeze())
        amounts = np.atleast_1d(eq.NP.values.squeeze())
        x_phase = np.atleast_2d(eq.X.values.squeeze())
        components = [str(c) for c in eq.component.values]

        fractions: dict[str, float] = {}
        f_gamma = 0.0
        c_gamma_weighted = 0.0
        for i, (ph, amt) in enumerate(zip(phase_names, amounts)):
            if not ph or np.isnan(amt):
                continue
            ph = str(ph)
            comp_i = {c.capitalize(): float(x_phase[i, j]) for j, c in enumerate(components)}
            # Distingue austenita (rica em Fe) de carbonitretos MX (FCC_A1#2)
            if ph == "FCC_A1" and comp_i.get("Fe", 0.0) < 0.5:
                ph = "MX"
            fractions[ph] = fractions.get(ph, 0.0) + float(amt)
            if ph == "FCC_A1":
                f_gamma += float(amt)
                c_gamma_weighted += float(amt) * mole_to_wt_fraction(comp_i).get("C", 0.0)

        undesired = sum(f for ph, f in fractions.items() if ph in UNDESIRED_PHASES)
        return EquilibriumResult(
            phase_fractions=fractions,
            austenite_fraction=f_gamma,
            austenite_carbon_wt=(c_gamma_weighted / f_gamma) if f_gamma > 1e-6 else None,
            undesired_fraction=undesired,
            converged=bool(fractions),
        )


# ---------------------------------------------------------------------------
# Amostragem
# ---------------------------------------------------------------------------
def latin_hypercube(
    bounds: Mapping[str, tuple[float, float]],
    n_samples: int,
    seed: int | None = None,
    log_features: frozenset[str] = LOG_FEATURES,
) -> pd.DataFrame:
    """Amostragem LHS otimizada (discrepância) dentro dos limites dados.

    Features em ``log_features`` são amostradas uniformemente em log10.
    Variáveis com limite inferior == superior são fixadas.
    """
    names = list(bounds)
    free = [n for n in names if bounds[n][1] > bounds[n][0]]
    out = pd.DataFrame(index=range(n_samples), columns=names, dtype=float)
    if free:
        sampler = qmc.LatinHypercube(d=len(free), seed=seed, optimization="random-cd")
        unit = sampler.random(n_samples)
        for j, name in enumerate(free):
            lo, hi = bounds[name]
            if name in log_features:
                out[name] = 10 ** (np.log10(lo) + unit[:, j] * (np.log10(hi) - np.log10(lo)))
            else:
                out[name] = lo + unit[:, j] * (hi - lo)
    for name in names:
        if name not in free:
            out[name] = bounds[name][0]
    return out


# ---------------------------------------------------------------------------
# Gerador
# ---------------------------------------------------------------------------
@dataclass
class NoiseModel:
    """Ruído experimental relativo (desvio-padrão) por propriedade."""

    ys_rel: float = 0.04
    uts_rel: float = 0.03
    el_rel: float = 0.08
    phase_abs: float = 0.01


class CalphadDataGenerator:
    """Gera datasets sintéticos composição/processo -> microestrutura -> propriedades.

    Cada liga (``alloy_id``) é processada em ``n_process_per_alloy`` rotas
    distintas; o ``alloy_id`` é o grupo usado no GroupKFold, evitando
    vazamento de informação composicional entre treino e validação.
    """

    def __init__(
        self,
        backend: ThermoBackend | None = None,
        composition_bounds: Mapping[str, tuple[float, float]] = COMPOSITION_BOUNDS,
        process_bounds: Mapping[str, tuple[float, float]] = PROCESS_BOUNDS,
        noise: NoiseModel | None = None,
        seed: int = 42,
    ) -> None:
        self.backend = backend or EmpiricalThermoBackend()
        self.composition_bounds = dict(composition_bounds)
        self.process_bounds = dict(process_bounds)
        self.noise = noise or NoiseModel()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample_compositions(self, n_alloys: int) -> pd.DataFrame:
        return latin_hypercube(self.composition_bounds, n_alloys, seed=self.seed)

    def sample_processes(self, n: int) -> pd.DataFrame:
        return latin_hypercube(self.process_bounds, n, seed=self.seed + 1)

    def generate(self, n_alloys: int = 500, n_process_per_alloy: int = 4) -> pd.DataFrame:
        """Gera o dataset completo."""
        comps = self.sample_compositions(n_alloys)
        comps.insert(0, "alloy_id", np.arange(n_alloys))
        procs = self.sample_processes(n_alloys * n_process_per_alloy)

        df = comps.loc[comps.index.repeat(n_process_per_alloy)].reset_index(drop=True)
        df = pd.concat([df, procs.reset_index(drop=True)], axis=1)

        logger.info("Calculando equilíbrio (%s) para %d pontos…", self.backend.name, len(df))
        eq = self.backend.equilibrium_batch(df[list(ELEMENTS)], df["T_anneal_C"].to_numpy())
        df = pd.concat([df, eq.reset_index(drop=True)], axis=1)
        df = df[df["eq_converged"] > 0].reset_index(drop=True)
        return self.compute_properties(df)

    def compute_properties(self, df: pd.DataFrame, add_noise: bool = True) -> pd.DataFrame:
        """Microestrutura + propriedades a partir de composição/processo/equilíbrio."""
        comp = {el: df[el].to_numpy(float) for el in ELEMENTS}
        c_gamma = df["C_austenite_eq_wt"].to_numpy(float) if "C_austenite_eq_wt" in df else None
        if c_gamma is not None and np.isnan(c_gamma).any():
            c_gamma = None  # fallback: balanço de massa
        micro = met.simulate_microstructure(
            comp,
            df["T_anneal_C"].to_numpy(float),
            df["cooling_rate_Cps"].to_numpy(float),
            austenite_fraction=df["f_austenite_eq"].to_numpy(float) if "f_austenite_eq" in df else None,
            austenite_carbon=c_gamma,
        )
        out = df.copy()
        for key, value in micro.items():
            out[key] = value

        phases = {p: out[p].to_numpy(float) for p in PHASES}
        d = out["grain_size_um"].to_numpy(float)
        ys, uts = met.rule_of_mixtures(comp, phases, d)
        ys, uts = np.asarray(ys, float), np.asarray(uts, float)
        el = np.asarray(met.elongation_model(uts, phases["f_retained_austenite"]), float)

        if add_noise:
            n = len(out)
            uts = uts * (1 + self.rng.normal(0, self.noise.uts_rel, n))
            ys = ys * (1 + self.rng.normal(0, self.noise.ys_rel, n))
            el = el * (1 + self.rng.normal(0, self.noise.el_rel, n))
            noisy = np.column_stack([out[p] for p in PHASES]) + self.rng.normal(
                0, self.noise.phase_abs, (n, len(PHASES)))
            noisy = np.clip(noisy, 0, None)
            noisy /= noisy.sum(axis=1, keepdims=True)
            for j, p in enumerate(PHASES):
                out[p] = noisy[:, j]

        out["YS_MPa"] = np.minimum(ys, 0.98 * uts)
        out["UTS_MPa"] = uts
        out["El_pct"] = np.clip(el, 1.0, None)
        out["UTSxEl_GPa_pct"] = out["UTS_MPa"] * out["El_pct"] / 1000.0
        out["thermo_backend"] = self.backend.name
        out["source"] = "synthetic"
        return out

    @staticmethod
    def save(df: pd.DataFrame, filename: str = "synthetic_ahss.csv",
             directory: Path = SYNTHETIC_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        df.to_csv(path, index=False)
        logger.info("Dataset salvo em %s (%d linhas)", path, len(df))
        return path


__all__ = [
    "CalphadDataGenerator", "EmpiricalThermoBackend", "EquilibriumResult", "NoiseModel",
    "PyCalphadBackend", "ThermoBackend", "latin_hypercube", "wt_to_mole_fraction",
    "mole_to_wt_fraction", "UNDESIRED_PHASES", "PROCESS_FEATURES", "TARGETS",
]
