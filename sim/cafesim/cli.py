"""
Command-line entry point. Run from inside sim/ with:

    uv run python -m cafesim.cli generate --config configs/base.yaml

Optional extras:
    --seed 7          use a different random seed than the config's
    --out ../data     choose where the output folder goes
"""
import argparse

from cafesim.config import load_config
from cafesim.engine import run
from cafesim.export import export


def main() -> None:
    # Set up the command and its options.
    parser = argparse.ArgumentParser(prog="cafesim")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a simulated dataset")
    gen.add_argument("--config", required=True, help="path to a YAML config")
    gen.add_argument("--seed", type=int, help="override the config's seed")
    gen.add_argument("--out", default="../data", help="output folder")

    # Read what the user typed.
    args = parser.parse_args()

    # Load and validate the settings.
    cfg = load_config(args.config)

    # If a seed was given on the command line, use it instead of the config's.
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})

    # Run the simulation, then save the results.
    raw, truth = run(cfg)
    raw_dir, truth_dir = export(cfg, raw, truth, args.out)

    # Print a short summary so you can see it worked.
    total = raw["hourly_customers"]["customers"].sum()
    print(f"done: {cfg.name}, seed {cfg.seed}, {cfg.days} days")
    print(f"customers: {total:,} total, {total / cfg.days:,.0f} per day on average")
    print(f"raw data   -> {raw_dir}")
    print(f"truth data -> {truth_dir}")


# This runs main() only when the file is run directly,
# not when another file imports it.
if __name__ == "__main__":
    main()