"""Interface de linha de comando do pipeline completo.

Exemplos::

    python -m src.cli generate --n-alloys 1000 --n-process 4
    python -m src.cli generate --tdb caminho/al_database.tdb --n-alloys 200
    python -m src.cli train --n-trials 20
    python -m src.cli optimize --min-uts 280 --min-el 5 --scrap tense_cast=0.7 wheels_a356=0.3 \\
        --min-recycled 0.6 --fix cooling_rate_Cps=2.0
    python -m src.cli charge --target Si=7 Fe=0.3 Cu=0.5 Mn=0.15 Mg=0.35 --objective co2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_PATH,
    INPUT_FEATURES,
    PROCESSED_DIR,
    SCRAP_GRADES,
    SYNTHETIC_DIR,
    OptimizationConfig,
    TrainingConfig,
)

logger = logging.getLogger("alloy-inverse-design")


def parse_key_values(items: list[str] | None, allowed: set[str] | None = None) -> dict[str, float]:
    """Converte ``["Si=7", "Fe=0.3"]`` em ``{"Si": 7.0, "Fe": 0.3}``."""
    out: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Esperado chave=valor, recebido: {item!r}")
        key, value = item.split("=", 1)
        if allowed is not None and key not in allowed:
            raise argparse.ArgumentTypeError(f"Chave desconhecida {key!r}; opções: {sorted(allowed)}")
        out[key] = float(value)
    return out


def cmd_generate(args: argparse.Namespace) -> None:
    from src.data_pipeline.calphad_gen import (
        CalphadDataGenerator,
        EmpiricalThermoBackend,
        PyCalphadBackend,
    )

    backend = PyCalphadBackend(args.tdb) if args.tdb else EmpiricalThermoBackend()
    gen = CalphadDataGenerator(backend=backend, seed=args.seed)
    df = gen.generate(n_alloys=args.n_alloys, n_process_per_alloy=args.n_process)
    gen.save(df, args.output)


def cmd_train(args: argparse.Namespace) -> None:
    from src.data_pipeline.preprocessing import preprocess
    from src.models.train import HyperParams, PINNTrainer

    frames = [pd.read_csv(p) for p in args.data]
    ds = preprocess(pd.concat(frames, ignore_index=True), save_dir=PROCESSED_DIR)
    cfg = TrainingConfig(n_trials=args.n_trials, max_epochs=args.epochs, n_splits=args.folds)
    trainer = PINNTrainer(cfg)
    res = trainer.run(ds, tune=args.n_trials > 0, params=HyperParams(), save_path=Path(args.model))
    print(res.cv_metrics.mean(numeric_only=True).round(3).to_string())


def cmd_optimize(args: argparse.Namespace) -> None:
    from src.models.pinn import load_checkpoint
    from src.optimization.inverse_design import InverseDesigner

    model, _ = load_checkpoint(args.model)
    cfg = OptimizationConfig(
        pop_size=args.pop_size, n_generations=args.generations,
        min_uts_MPa=args.min_uts, min_elongation_pct=args.min_el, min_ys_MPa=args.min_ys,
        min_recycled_fraction=args.min_recycled,
        scrap_mix=parse_key_values(args.scrap, set(SCRAP_GRADES)) or {"tense_cast": 1.0},
        low_carbon_primary=args.low_carbon_primary,
        fixed_values=parse_key_values(args.fix, set(INPUT_FEATURES)),
        n_mc_samples=args.mc_samples, max_uts_cv=args.max_uts_cv,
    )
    res = InverseDesigner(model, cfg).run(verbose=args.verbose)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.pareto.to_csv(out, index=False)
    print(f"{len(res.pareto)} soluções de Pareto (viável={res.feasible}) -> {out}")
    print(InverseDesigner.select_compromise(res.pareto).to_string())


def cmd_charge(args: argparse.Namespace) -> None:
    from src.optimization.charge import optimize_charge_lp

    target = parse_key_values(args.target)
    res = optimize_charge_lp(target, objective=args.objective, co2_weight=args.co2_price,
                             low_carbon_primary=args.low_carbon_primary,
                             tolerance_wt=parse_key_values(args.tolerance))
    if not res.success:
        print(f"Carga inviável: {res.message}")
        return
    for name, frac in sorted(res.fractions.items(), key=lambda kv: -kv[1]):
        if frac > 1e-5:
            print(f"{name:28s} {1000 * frac:8.1f} kg/t")
    print(f"Sucata: {100 * res.scrap_fraction:.1f} % · custo {res.cost_USD_per_kg:.3f} USD/kg · "
          f"CO2 {res.CO2_kg_per_kg:.2f} kg/kg")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alloy-inverse-design")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Gera dados sintéticos (CALPHAD + LHS)")
    g.add_argument("--n-alloys", type=int, default=1000)
    g.add_argument("--n-process", type=int, default=4)
    g.add_argument("--tdb", type=str, default=None, help="Base TDB de Al (ativa o backend pycalphad)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", type=str, default=DEFAULT_DATASET_NAME)
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Treina a PINN (Optuna + GroupKFold)")
    t.add_argument("--data", nargs="+", default=[str(SYNTHETIC_DIR / DEFAULT_DATASET_NAME)])
    t.add_argument("--n-trials", type=int, default=20)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    t.set_defaults(func=cmd_train)

    o = sub.add_parser("optimize", help="Design inverso NSGA-III")
    o.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH))
    o.add_argument("--min-uts", type=float, default=260.0)
    o.add_argument("--min-el", type=float, default=4.0)
    o.add_argument("--min-ys", type=float, default=0.0)
    o.add_argument("--min-recycled", type=float, default=0.0, help="Fração mínima de sucata (0–1)")
    o.add_argument("--scrap", nargs="*", metavar="CLASSE=PROPORÇÃO",
                   help=f"Mistura de sucata; classes: {', '.join(SCRAP_GRADES)}")
    o.add_argument("--fix", nargs="*", metavar="VAR=VALOR",
                   help="Fixa variáveis (ex.: cooling_rate_Cps=2 Ni=0)")
    o.add_argument("--low-carbon-primary", action="store_true", help="Al primário hidroelétrico")
    o.add_argument("--mc-samples", type=int, default=0)
    o.add_argument("--max-uts-cv", type=float, default=None)
    o.add_argument("--pop-size", type=int, default=120)
    o.add_argument("--generations", type=int, default=150)
    o.add_argument("--output", type=str, default=str(PROCESSED_DIR / "pareto_front.csv"))
    o.set_defaults(func=cmd_optimize)

    c = sub.add_parser("charge", help="Carga ótima multi-sucata (programação linear)")
    c.add_argument("--target", nargs="+", required=True, metavar="EL=WT%")
    c.add_argument("--tolerance", nargs="*", metavar="EL=WT%", help="Faixa ± por elemento")
    c.add_argument("--objective", choices=["cost", "co2", "weighted"], default="cost")
    c.add_argument("--co2-price", type=float, default=0.0, help="USD/kg CO2 (objetivo weighted)")
    c.add_argument("--low-carbon-primary", action="store_true")
    c.set_defaults(func=cmd_charge)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
