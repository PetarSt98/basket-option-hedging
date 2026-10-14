"""Local fixture generation and replay commands; no trading connectivity."""

import argparse
from pathlib import Path
import sys

from .contracts import InputError
from .deterministic import write_deterministic_cases
from .engine import Engine
from .policies import ScheduledDeltaPolicy, UnhedgedPolicy, stock_delta_hedge
from .replay import load_case
from .risk import UnnettedLiabilityStressMargin


def main(argv: list[str] | None = None) -> int:
    # Keep optional historical dependencies out of the original synthetic CLI.
    provided = list(sys.argv[1:] if argv is None else argv)
    if provided and provided[0] == "historical":
        from .historical_cli import main as historical_main
        return historical_main(provided[1:])
    parser = argparse.ArgumentParser(prog="basket-hedging", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    deterministic = commands.add_parser("make-deterministic", help="write the small hand-authored fixtures")
    deterministic.add_argument("--output-dir", type=Path, default=Path("fixtures"))
    generate = commands.add_parser("generate", help="save a synthetic GBM replay and point-in-time valuations")
    generate.add_argument("--output", type=Path, default=Path("fixtures/synthetic.json"))
    generate.add_argument("--replay-seed", type=int, default=20260912)
    generate.add_argument("--pricing-seed", type=int, default=731)
    generate.add_argument("--samples", type=int, default=8192)
    generate.add_argument("--constituents", type=int, default=3)
    generate.add_argument("--days", type=int, default=20)
    generate.add_argument("--basket-kind", choices=("put", "call"), default="put")
    run = commands.add_parser("run", help="stream a saved case through one or both baselines")
    run.add_argument("--case", type=Path, required=True)
    run.add_argument("--policy", choices=("unhedged", "delta", "both"), default="both")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--every-update", action="store_true",
                     help="evaluate the same delta rule at every clock update instead of scheduled reviews")
    comparison = commands.add_parser("compare-scenarios", help="compare four policies using synthetic conditional scenarios")
    comparison.add_argument("--case", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--scenarios", type=int, default=256)
    comparison.add_argument("--scenario-seed", type=int, default=9173)
    comparison.add_argument("--pricing-samples", type=int, default=1024)
    comparison.add_argument("--alpha", type=float, default=.1)
    comparison.add_argument("--cost-proxy", type=float, default=.02,
                            help="currency per share/contract; Python API also accepts instrument-specific weights")
    stage2 = commands.add_parser("stage2", help="train and validate a small learned synthetic scenario generator")
    stage2.add_argument("--output", type=Path, default=Path("reports/stage2"))
    stage2.add_argument("--test-paths", type=int, default=200)
    stage2.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "make-deterministic":
            for kind, path in write_deterministic_cases(arguments.output_dir).items():
                print(f"{kind}: {path}")
        elif arguments.command == "generate":
            from .synthetic import generate_synthetic_case

            path = generate_synthetic_case(arguments.output, replay_seed=arguments.replay_seed,
                                           pricing_seed=arguments.pricing_seed, samples=arguments.samples,
                                           constituents=arguments.constituents, days=arguments.days,
                                           basket_kind=arguments.basket_kind)
            print(f"Synthetic fixture written: {path}")
        elif arguments.command == "stage2":
            from .stage2 import run_stage2

            summary = run_stage2(arguments.output, test_paths=arguments.test_paths, workers=arguments.workers)
            for outcome in summary["outcomes"]:
                print(f"{outcome['policy']}: hedge-error SD={outcome['hedge_error_sd']}; "
                      f"completion={outcome['completion_frequency']:.1%}")
            print(f"Reports: {arguments.output}")
        elif arguments.command == "compare-scenarios":
            from .scenario_comparison import run_scenario_comparison

            summary = run_scenario_comparison(
                arguments.case, arguments.output, samples=arguments.scenarios,
                scenario_seed=arguments.scenario_seed, pricing_samples=arguments.pricing_samples,
                alpha=arguments.alpha, cost_proxy=arguments.cost_proxy)
            for outcome in summary["outcomes"]:
                print(f"{outcome['policy']}: {outcome['status']}; observed marked P&L={outcome['net_pnl']}")
            print(f"Reports: {arguments.output}")
        else:
            from .reporting import write_reports

            case = load_case(arguments.case)
            delta = stock_delta_hedge if arguments.every_update else ScheduledDeltaPolicy()
            policies = ([UnhedgedPolicy(), delta] if arguments.policy == "both" else
                        [UnhedgedPolicy()] if arguments.policy == "unhedged" else [delta])
            results = [Engine(case.instruments, case.opening, case.rules,
                              UnnettedLiabilityStressMargin(case.rules.expected_stress_scenarios))
                       .run(case.events(), policy) for policy in policies]
            comparison = write_reports(results, arguments.output, case_path=arguments.case, case=case)
            for outcome in comparison["outcomes"]:
                pnl = outcome["marked_pnl"]
                displayed_pnl = "unknown" if pnl is None else f"{pnl:.6f}"
                print(f"{outcome['policy']}: {outcome['status']}; observed marked P&L={displayed_pnl}")
            print(f"Reports: {arguments.output}")
    except (InputError, OSError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 2
    # A recorded financial breach is an experimental outcome, not a CLI malfunction.
    return 0
