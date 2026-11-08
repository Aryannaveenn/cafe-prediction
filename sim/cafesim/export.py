"""
Saves the simulation results to disk as Parquet files.

Parquet is a compact, fast file format for tables, and keeps column types
(dates stay dates, numbers stay numbers), unlike CSV.

Output layout (inside the repo's data/ folder):
  data/raw/<scenario>_seed<N>/     <- what the model is allowed to see
  data/truth/<scenario>_seed<N>/   <- hidden answers, for scoring only
"""
from pathlib import Path

import polars as pl

from cafesim.config import SimConfig
from cafesim.truth import TruthLog


def export(
    cfg: SimConfig, raw: dict[str, pl.DataFrame], truth: TruthLog, out_dir: str | Path
) -> tuple[Path, Path]:
    # Name each run after its scenario and seed, e.g. "base_seed42",
    # so different runs never overwrite each other.
    run_name = f"{cfg.name}_seed{cfg.seed}"

    raw_dir = Path(out_dir) / "raw" / run_name
    truth_dir = Path(out_dir) / "truth" / run_name

    # Create the folders if they don't exist yet.
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    # Save each observable table, e.g. raw/base_seed42/hourly_customers.parquet
    for name, df in raw.items():
        df.write_parquet(raw_dir / f"{name}.parquet")

    # Save each ground-truth table.
    for name, df in truth.tables.items():
        df.write_parquet(truth_dir / f"{name}.parquet")

    # The config itself reveals the true settings (e.g. the real holiday
    # effect), so it's saved with the truth, never with the raw data.
    (truth_dir / "config.json").write_text(cfg.model_dump_json(indent=2))

    return raw_dir, truth_dir