"""Interface de linha de comando do pipeline completo.

Exemplos::

    python -m src.cli generate --n-alloys 1000 --n-process 4
    python -m src.cli generate --tdb caminho/mc_fe.tdb --n-alloys 200
    python -m src.cli train --n-trials 20
    python -m src.cli optimize --min-uts 1180 --min-el 10 --generations 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, SYNTHETIC_DIR, OptimizationConfig, TrainingConfig

logger = logging.getLogger("alloy-inverse-design")


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
        min_uts_MPa=args.min_uts, min_elongation_pct=args.min_el, ceq_max=args.ceq_max,
        fe_route=args.fe_route,
    )
    res = InverseDesigner(model, cfg).run(verbose=args.verbose)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.pareto.to_csv(out, index=False)
    print(f"{len(res.pareto)} soluções de Pareto (viável={res.feasible}) -> {out}")
    print(InverseDesigner.select_compromise(res.pareto).to_string())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alloy-inverse-design")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Gera dados sintéticos (CALPHAD + LHS)")
    g.add_argument("--n-alloys", type=int, default=1000)
    g.add_argument("--n-process", type=int, default=4)
    g.add_argument("--tdb", type=str, default=None, help="Base TDB (ativa o backend pycalphad)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output", type=str, default="synthetic_ahss.csv")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Treina a PINN (Optuna + GroupKFold)")
    t.add_argument("--data", nargs="+", default=[str(SYNTHETIC_DIR / "synthetic_ahss.csv")])
    t.add_argument("--n-trials", type=int, default=20)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--model", type=str, default=str(MODELS_DIR / "pinn_ahss.pt"))
    t.set_defaults(func=cmd_train)

    o = sub.add_parser("optimize", help="Design inverso NSGA-III")
    o.add_argument("--model", type=str, default=str(MODELS_DIR / "pinn_ahss.pt"))
    o.add_argument("--min-uts", type=float, default=980.0)
    o.add_argument("--min-el", type=float, default=12.0)
    o.add_argument("--ceq-max", type=float, default=0.60)
    o.add_argument("--pop-size", type=int, default=120)
    o.add_argument("--generations", type=int, default=150)
    o.add_argument("--fe-route", choices=["BF-BOF", "EAF"], default="BF-BOF")
    o.add_argument("--output", type=str, default=str(PROCESSED_DIR / "pareto_front.csv"))
    o.set_defaults(func=cmd_optimize)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
