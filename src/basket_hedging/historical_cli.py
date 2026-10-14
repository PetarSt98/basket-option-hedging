"""Explicit data, training and backtest steps for the historical adaptation."""

import argparse
from pathlib import Path

from .contracts import DataGap, InputError
from .replay import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="basket-hedging historical", description=__doc__)
    steps = parser.add_subparsers(dest="step", required=True)
    fetch = steps.add_parser("fetch", help="download exactly six pinned, hash-checked release files")
    fetch.add_argument("--raw", required=True, type=Path)
    check = steps.add_parser("check", help="inspect schemas, common dates and corporate-action completeness")
    check.add_argument("--raw", required=True, type=Path)
    check.add_argument("--output", required=True, type=Path)
    prepare = steps.add_parser("prepare", help="build daily joint surfaces and immutable provenance")
    prepare.add_argument("--raw", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    supplement = steps.add_parser("supplement", help="rebuild a separate prepared copy with verified IWM actions")
    supplement.add_argument("--prepared", required=True, type=Path)
    supplement.add_argument("--actions", type=Path, default=Path("data/iwm_corporate_actions"))
    supplement.add_argument("--output", required=True, type=Path)
    train = steps.add_parser("train", help="train historical models on the predeclared chronological split")
    train.add_argument("--prepared", required=True, type=Path)
    train.add_argument("--output", type=Path, default=Path("reports/historical/models"))
    train.add_argument("--model", choices=("gaussian", "volgan", "both"), default="both")
    run = steps.add_parser("run", help="compare five policies; unknown corporate actions block financial results")
    run.add_argument("--prepared", required=True, type=Path)
    run.add_argument("--models", type=Path, default=Path("reports/historical/models"))
    run.add_argument("--output", type=Path, default=Path("reports/historical/benchmark"))
    figures = steps.add_parser("figures", help="draw four static figures from saved historical results")
    figures.add_argument("--report", type=Path, default=Path("reports/historical/benchmark"))
    args = parser.parse_args(argv)
    try:
        if args.step == "fetch":
            from .etf_data import download_release
            result = download_release(args.raw)
            print(result)
        elif args.step == "check":
            from .etf_data import check_data_readiness
            result = check_data_readiness(args.raw)
            write_json(args.output, result)
            print(f"Data inspection: {args.output}")
            print(f"Financial benchmark ready: {result['benchmark_ready']}")
        elif args.step == "prepare":
            from .etf_data import prepare_etf_data
            result = prepare_etf_data(args.raw, args.output)
            print(f"Prepared: {args.output}; SHA-256 {result['prepared_sha256']}")
        elif args.step == "supplement":
            from .historical_actions import supplement_prepared
            result = supplement_prepared(args.prepared, args.actions, args.output)
            print(f"Supplemented: {args.output}; SHA-256 {result['prepared_sha256']}")
        elif args.step == "train":
            args.output.mkdir(parents=True, exist_ok=True)
            if args.model in {"gaussian", "both"}:
                from .historical_training import train_historical_gaussian
                model = train_historical_gaussian(args.prepared, args.output / "gaussian_training")
                model.save(args.output / "gaussian.json")
            if args.model in {"volgan", "both"}:
                from .volgan import train_prepared_volgan
                train_prepared_volgan(args.prepared, args.output / "volgan.json")
        elif args.step == "run":
            from .historical import run_historical
            from .historical_report import paired_report
            result = run_historical(args.prepared, args.models, args.output)
            paired_report(args.output)
            for row in result["metrics"]:
                print(f"{row['policy']}: hedge-error SD={row['hedge_error_sd']}; "
                      f"completed={row['completion_rate']:.1%}; breaches={row['margin_breaches']}")
        else:
            from .historical_plots import draw_figures
            for path in draw_figures(args.report):
                print(path)
    except DataGap as error:
        if args.step == "run":
            write_json(args.output / "blocked.json", dict(status="BLOCKED", reason=str(error),
                financial_results_generated=False, substitute_dataset_used=False))
        print(f"BLOCKED: {error}")
        return 2
    except (InputError, OSError) as error:
        print(f"Input error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
