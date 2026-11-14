"""Geração de dados sintéticos via CALPHAD + Latin Hypercube Sampling.

Fluxo::

    LHS (composição × processo)
      -> equilíbrio termodinâmico em DUAS temperaturas:
           * T de solubilização: solutos dissolvidos (Cu, Mg), fases residuais,
             líquido (fusão incipiente);
           * T do forno de espera: intermetálicos primários de Fe/Mn/Cr (sludge);
      -> modelo de solidificação/T6 (``src.metallurgy``)
      -> propriedades mecânicas + ruído experimental.

Backends termodinâmicos:

* :class:`PyCalphadBackend` — energia de Gibbs mínima com uma base TDB de
  alumínio (ex.: COST 507, TCAL, PanAl). Lento (~s/ponto): use para
  datasets de validação ou centenas de pontos.
* :class:`EmpiricalThermoBackend` — solvus de Arrhenius + balanço
  estequiométrico, vetorizado. Instantâneo; útil para prototipagem e testes,
  mas NÃO substitui CALPHAD para conclusões quantitativas.
"""

from __future__ import annotations

import logging
import re
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
    DEFAULT_DATASET_NAME,
    ELEMENTS,
    KELVIN_OFFSET,
    LOG_FEATURES,
    PHASES,
    PRESSURE_PA,
    PROCESS_BOUNDS,
    SYNTHETIC_DIR,
    MetallurgicalLimits,
)

logger = logging.getLogger(__name__)

EQ_COLUMNS: tuple[str, ...] = (
    "f_liquid_eq", "Cu_fcc_eq_wt", "Mg_fcc_eq_wt", "f_Al2Cu_eq", "f_Mg2Si_eq",
    "f_beta_eq", "f_alpha_eq", "f_Si_eq", "f_sludge_eq", "f_undesired_eq", "eq_converged",
)

# Classificação de fases CALPHAD por padrão de nome (bases COST507/TCAL/PanAl variam).
PHASE_PATTERNS: dict[str, tuple[str, ...]] = {
    "liquid": (r"^LIQUID",),
    "Si": (r"^DIAMOND", r"^SI$"),
    "Al2Cu": (r"AL2CU", r"THETA"),
    "Mg2Si": (r"MG2SI",),
    "beta_AlFeSi": (r"AL5FESI", r"BETA_ALFESI", r"ALFESI_BETA", r"AL9FE2SI2", r"TAU6"),
    "alpha_AlFeMnSi": (r"AL15SI2M4", r"ALPHA_ALFE", r"ALFEMNSI", r"AL19FE4MNSI2", r"AL15.*MN", r"TAU5"),
    "Fe_other": (r"AL13FE4", r"AL3FE", r"AL6MN", r"AL4MN", r"AL7CR", r"AL45CR7", r"AL3NI", r"AL3TI"),
}


def classify_phase(name: str) -> str:
    """Mapeia um nome de fase CALPHAD para uma classe metalúrgica."""
    upper = name.upper().split("#")[0]
    for cls, patterns in PHASE_PATTERNS.items():
        if any(re.search(p, upper) for p in patterns):
            return cls
    if upper.startswith("FCC_A1"):
        return "fcc"
    return "other"


# ---------------------------------------------------------------------------
# Conversões
# ---------------------------------------------------------------------------
def wt_to_mole_fraction(composition: Mapping[str, float]) -> dict[str, float]:
    """Converte wt% (Al = balanço) para fração molar."""
    wt = {el: float(composition.get(el, 0.0)) for el in ELEMENTS}
    wt["Al"] = 100.0 - sum(wt.values())
    if wt["Al"] <= 0:
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
    matrix_composition_wt: dict[str, float] = field(default_factory=dict)
    converged: bool = True

    def fraction(self, cls: str) -> float:
        return float(self.phase_fractions.get(cls, 0.0))


class ThermoBackend(ABC):
    """Interface de um backend termodinâmico para ligas de Al."""

    name: str = "abstract"

    def __init__(self, holding_temperature_C: float | None = None) -> None:
        self.holding_temperature_C = (
            holding_temperature_C if holding_temperature_C is not None
            else MetallurgicalLimits().holding_temperature_C
        )

    @abstractmethod
    def equilibrium(self, composition: Mapping[str, float], T_C: float) -> EquilibriumResult:
        """Equilíbrio para uma liga em uma temperatura (°C), frações agrupadas por classe."""

    def solution_and_holding(self, composition: Mapping[str, float], T_solution_C: float) -> dict[str, float]:
        """Variáveis termodinâmicas usadas pelo gerador e pelo filtro."""
        sol = self.equilibrium(composition, T_solution_C)
        hold = self.equilibrium(composition, self.holding_temperature_C)
        sludge = sum(hold.fraction(c) for c in ("beta_AlFeSi", "alpha_AlFeMnSi", "Fe_other"))
        return {
            "f_liquid_eq": sol.fraction("liquid"),
            "Cu_fcc_eq_wt": sol.matrix_composition_wt.get("Cu", 0.0),
            "Mg_fcc_eq_wt": sol.matrix_composition_wt.get("Mg", 0.0),
            "f_Al2Cu_eq": sol.fraction("Al2Cu"),
            "f_Mg2Si_eq": sol.fraction("Mg2Si"),
            "f_beta_eq": sol.fraction("beta_AlFeSi"),
            "f_alpha_eq": sol.fraction("alpha_AlFeMnSi"),
            "f_Si_eq": sol.fraction("Si"),
            "f_sludge_eq": sludge,
            "f_undesired_eq": sludge,
            "eq_converged": float(sol.converged and hold.converged),
        }

    def equilibrium_batch(self, compositions: pd.DataFrame, T_solution_C: np.ndarray) -> pd.DataFrame:
        """Equilíbrio para várias ligas. Implementação padrão: laço tolerante a falhas."""
        rows: list[dict[str, float]] = []
        for (_, comp), t in zip(compositions.iterrows(), np.asarray(T_solution_C, dtype=float)):
            try:
                rows.append(self.solution_and_holding(comp.to_dict(), float(t)))
            except Exception as exc:  # noqa: BLE001 — falhas de convergência
                logger.warning("Equilíbrio falhou (%s): %s", self.name, exc)
                rows.append({c: np.nan for c in EQ_COLUMNS} | {"eq_converged": 0.0})
        return pd.DataFrame(rows, index=compositions.index)[list(EQ_COLUMNS)]


class EmpiricalThermoBackend(ThermoBackend):
    """Aproximação vetorizada do equilíbrio (solvus + estequiometria + sludge factor)."""

    name = "empirical"

    def equilibrium(self, composition: Mapping[str, float], T_C: float) -> EquilibriumResult:
        comp = {el: float(composition.get(el, 0.0)) for el in ELEMENTS}
        t_inc = float(met.incipient_melting_temperature(comp))
        t_liq = float(met.liquidus_temperature(comp))
        if T_C > t_inc + 30.0:  # banho (semi)líquido: intermetálicos primários = sludge
            sf_excess = max(float(met.sludge_factor(comp)) - float(met.critical_sludge_factor(T_C)), 0.0)
            f_sludge = sf_excess / 100.0 / 0.30
            return EquilibriumResult({"liquid": 1.0 - f_sludge, "alpha_AlFeMnSi": f_sludge})
        cu_d, mg_d = (float(v) for v in met.dissolved_solutes(comp, T_C))
        ph = {k: float(v) for k, v in met.phase_fractions(comp, 1.0, T_C, cu_dissolved=cu_d).items()}
        f_liq = float(np.clip((T_C - t_inc) / max(t_liq - t_inc, 1.0), 0.0, 1.0))
        solid = 1.0 - f_liq
        fractions = {
            "fcc": ph["f_alpha_Al"] * solid,
            "Si": ph["f_Si_eutectic"] * solid,
            "beta_AlFeSi": ph["f_beta_AlFeSi"] * solid,
            "alpha_AlFeMnSi": ph["f_alpha_AlFeMnSi"] * solid,
            "Al2Cu": ph["f_Al2Cu"] * solid,
            "Mg2Si": max(comp["Mg"] - mg_d, 0.0) / 100.0 / 0.634,
            "liquid": f_liq,
        }
        return EquilibriumResult(fractions, {"Cu": cu_d, "Mg": mg_d})

    def solution_and_holding(self, composition: Mapping[str, float], T_solution_C: float) -> dict[str, float]:
        row = pd.DataFrame([{el: float(composition.get(el, 0.0)) for el in ELEMENTS}])
        return self.equilibrium_batch(row, np.array([T_solution_C])).iloc[0].to_dict()

    def equilibrium_batch(self, compositions: pd.DataFrame, T_solution_C: np.ndarray) -> pd.DataFrame:
        comp = {el: compositions[el].to_numpy(float) if el in compositions else np.zeros(len(compositions))
                for el in ELEMENTS}
        T = np.asarray(T_solution_C, dtype=float)
        cu_d, mg_d = (np.asarray(v, float) for v in met.dissolved_solutes(comp, T))
        ph = {k: np.asarray(v, float) for k, v in met.phase_fractions(comp, 1.0, T, cu_dissolved=cu_d).items()}
        t_inc = np.asarray(met.incipient_melting_temperature(comp), float)
        t_liq = np.asarray(met.liquidus_temperature(comp), float)
        f_liq = np.clip((T - t_inc) / np.clip(t_liq - t_inc, 1.0, None), 0.0, 1.0)
        sf = np.asarray(met.sludge_factor(comp), float)
        sf_crit = np.asarray(met.critical_sludge_factor(self.holding_temperature_C), float)
        sludge = np.clip(sf - sf_crit, 0.0, None) / 100.0 / 0.30
        ones = np.ones(len(T))
        return pd.DataFrame(
            {
                "f_liquid_eq": f_liq,
                "Cu_fcc_eq_wt": cu_d,
                "Mg_fcc_eq_wt": mg_d,
                "f_Al2Cu_eq": ph["f_Al2Cu"] * (1 - f_liq),
                "f_Mg2Si_eq": np.clip(comp["Mg"] - mg_d, 0.0, None) / 100.0 / 0.634,
                "f_beta_eq": ph["f_beta_AlFeSi"] * (1 - f_liq),
                "f_alpha_eq": ph["f_alpha_AlFeMnSi"] * (1 - f_liq),
                "f_Si_eq": ph["f_Si_eutectic"] * (1 - f_liq),
                "f_sludge_eq": sludge,
                "f_undesired_eq": sludge,
                "eq_converged": ones,
            },
            index=compositions.index,
        )


class PyCalphadBackend(ThermoBackend):
    """Equilíbrio via pycalphad com uma base TDB de alumínio.

    Elementos ausentes da base são ignorados (com aviso único). As fases são
    agrupadas em classes metalúrgicas por :func:`classify_phase`; a matriz é
    a FCC_A1 rica em Al, cuja composição fornece o Cu/Mg dissolvidos.

    Parameters
    ----------
    tdb_path:
        Caminho para a base TDB.
    phases:
        Fases a considerar. Padrão: todas as fases da base.
    min_mole_fraction:
        Solutos abaixo deste valor são descartados (estabilidade numérica).
    """

    name = "pycalphad"

    def __init__(
        self,
        tdb_path: str | Path,
        phases: Sequence[str] | None = None,
        min_mole_fraction: float = 1e-5,
        holding_temperature_C: float | None = None,
    ) -> None:
        super().__init__(holding_temperature_C)
        try:
            from pycalphad import Database
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pycalphad não instalado: pip install pycalphad") from exc
        self.db = Database(str(tdb_path))
        self.min_mole_fraction = min_mole_fraction
        db_phases = sorted(self.db.phases.keys())
        self.phases = [p for p in (phases or db_phases) if p in db_phases]
        if not self.phases:
            raise ValueError("Nenhuma das fases solicitadas existe na base TDB.")
        self.db_elements = {str(e).upper() for e in self.db.elements}
        if "AL" not in self.db_elements:
            raise ValueError("A base TDB não contém Al.")
        self._warned: set[str] = set()

    def equilibrium(self, composition: Mapping[str, float], T_C: float) -> EquilibriumResult:
        from pycalphad import equilibrium, variables as v

        x = wt_to_mole_fraction(composition)
        solutes = []
        for el, xi in x.items():
            if el == "Al" or xi < self.min_mole_fraction:
                continue
            if el.upper() not in self.db_elements:
                if el not in self._warned:
                    logger.warning("Elemento %s ausente da base TDB — ignorado.", el)
                    self._warned.add(el)
                continue
            solutes.append(el)
        comps = ["AL", "VA"] + [el.upper() for el in solutes]
        conds: dict = {v.T: T_C + KELVIN_OFFSET, v.P: PRESSURE_PA, v.N: 1}
        for el in solutes:
            conds[v.X(el.upper())] = x[el]

        eq = equilibrium(self.db, comps, self.phases, conds)
        phase_names = np.atleast_1d(eq.Phase.values.squeeze())
        amounts = np.atleast_1d(eq.NP.values.squeeze())
        x_phase = np.atleast_2d(eq.X.values.squeeze())
        components = [str(c) for c in eq.component.values]

        fractions: dict[str, float] = {}
        matrix: dict[str, float] = {}
        matrix_amount = 0.0
        for i, (ph, amt) in enumerate(zip(phase_names, amounts)):
            if not ph or np.isnan(amt) or amt <= 0:
                continue
            comp_i = {c.capitalize(): float(x_phase[i, j]) for j, c in enumerate(components)}
            cls = classify_phase(str(ph))
            if cls == "fcc" and comp_i.get("Al", 0.0) < 0.5:
                cls = "other"  # FCC não-matriz (ex.: Cu-rico)
            fractions[cls] = fractions.get(cls, 0.0) + float(amt)
            if cls == "fcc" and float(amt) > matrix_amount:
                matrix_amount = float(amt)
                matrix = mole_to_wt_fraction(comp_i)
        # NP é fração molar de fase; para ligas diluídas ≈ fração mássica.
        return EquilibriumResult(fractions, matrix, converged=bool(fractions))


# ---------------------------------------------------------------------------
# Amostragem
# ---------------------------------------------------------------------------
def latin_hypercube(
    bounds: Mapping[str, tuple[float, float]],
    n_samples: int,
    seed: int | None = None,
    log_features: frozenset[str] = LOG_FEATURES,
) -> pd.DataFrame:
    """Amostragem LHS otimizada (discrepância centrada) dentro dos limites.

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
    """Ruído experimental (desvio-padrão relativo) por propriedade.

    Fundidos têm dispersão de ductilidade alta (porosidade, óxidos/bifilmes).
    """

    ys_rel: float = 0.04
    uts_rel: float = 0.04
    el_rel: float = 0.12
    phase_rel: float = 0.05
    phase_abs: float = 0.001


class CalphadDataGenerator:
    """Gera datasets composição/processo -> microestrutura -> propriedades.

    Cada liga (``alloy_id``) é processada em ``n_process_per_alloy`` rotas
    distintas (molde/solubilização/envelhecimento); o ``alloy_id`` é o grupo
    usado no GroupKFold, evitando vazamento composicional entre treino e validação.
    """

    def __init__(
        self,
        backend: ThermoBackend | None = None,
        composition_bounds: Mapping[str, tuple[float, float]] = COMPOSITION_BOUNDS,
        process_bounds: Mapping[str, tuple[float, float]] = PROCESS_BOUNDS,
        noise: NoiseModel | None = None,
        seed: int = 42,
        solution_margin_C: tuple[float, float] | None = (-15.0, 75.0),
    ) -> None:
        """
        ``solution_margin_C``: se definido, a T de solubilização é amostrada como
        ``T_incipiente − margem`` (margem via LHS neste intervalo), imitando a
        prática industrial; margens negativas geram exemplos "queimados" para que
        o modelo aprenda o dano. ``None`` amostra T_solution uniformemente.
        """
        self.solution_margin_C = solution_margin_C
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
        if self.solution_margin_C is not None:
            lo, hi = self.process_bounds["T_solution_C"]
            u = (df["T_solution_C"] - lo) / max(hi - lo, 1e-9)  # estrato LHS reaproveitado
            m_lo, m_hi = self.solution_margin_C
            t_inc = np.asarray(met.incipient_melting_temperature({el: df[el] for el in ELEMENTS}), float)
            df["T_solution_C"] = np.clip(t_inc - (m_lo + u * (m_hi - m_lo)), lo, hi)

        logger.info("Calculando equilíbrio (%s) para %d pontos…", self.backend.name, len(df))
        eq = self.backend.equilibrium_batch(df[list(ELEMENTS)], df["T_solution_C"].to_numpy())
        df = pd.concat([df, eq.reset_index(drop=True)], axis=1)
        df = df[df["eq_converged"] > 0].reset_index(drop=True)
        return self.compute_properties(df)

    def compute_properties(self, df: pd.DataFrame, add_noise: bool = True) -> pd.DataFrame:
        """Microestrutura + propriedades a partir de composição/processo/equilíbrio."""
        comp = {el: df[el].to_numpy(float) for el in ELEMENTS}

        def _eq(col: str) -> np.ndarray | None:
            if col not in df:
                return None
            values = df[col].to_numpy(float)
            return None if np.isnan(values).any() else values

        sim = met.simulate_casting(
            comp,
            df["cooling_rate_Cps"].to_numpy(float),
            df["T_solution_C"].to_numpy(float),
            df["T_aging_C"].to_numpy(float),
            df["t_aging_h"].to_numpy(float),
            cu_dissolved=_eq("Cu_fcc_eq_wt"),
            mg_dissolved=_eq("Mg_fcc_eq_wt"),
        )
        out = df.copy()
        for key, value in sim.items():
            out[key] = value

        if add_noise:
            n = len(out)
            nz = self.noise
            out["YS_MPa"] *= 1 + self.rng.normal(0, nz.ys_rel, n)
            out["UTS_MPa"] *= 1 + self.rng.normal(0, nz.uts_rel, n)
            out["El_pct"] *= np.exp(self.rng.normal(0, nz.el_rel, n))  # log-normal: El > 0
            minor = [p for p in PHASES if p != "f_alpha_Al"]
            noisy = out[minor].to_numpy(float) * (1 + self.rng.normal(0, nz.phase_rel, (n, len(minor))))
            noisy = np.clip(noisy + self.rng.normal(0, nz.phase_abs, noisy.shape), 0.0, None)
            noisy *= (out[minor].to_numpy(float) > 0)  # fases ausentes continuam ausentes
            out[minor] = noisy
            out["f_alpha_Al"] = 1.0 - noisy.sum(axis=1)

        out["UTS_MPa"] = np.maximum(out["UTS_MPa"], out["YS_MPa"] * 1.01)
        out["El_pct"] = out["El_pct"].clip(lower=0.2)
        out["quality_index_MPa"] = out["UTS_MPa"] + 150.0 * np.log10(out["El_pct"])
        out["thermo_backend"] = self.backend.name
        out["source"] = "synthetic"
        return out

    @staticmethod
    def save(df: pd.DataFrame, filename: str = DEFAULT_DATASET_NAME,
             directory: Path = SYNTHETIC_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        df.to_csv(path, index=False)
        logger.info("Dataset salvo em %s (%d linhas)", path, len(df))
        return path


__all__ = [
    "CalphadDataGenerator", "EmpiricalThermoBackend", "EquilibriumResult", "NoiseModel",
    "PyCalphadBackend", "ThermoBackend", "classify_phase", "latin_hypercube",
    "wt_to_mole_fraction", "mole_to_wt_fraction", "EQ_COLUMNS",
]
