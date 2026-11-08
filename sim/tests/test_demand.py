"""
Automatic checks that the demand simulation behaves correctly.
Run them from inside sim/ with:  uv run pytest
"""
from pathlib import Path

from cafesim.config import load_config
from cafesim.engine import run

# Path to the base config, worked out relative to this test file so the
# tests work no matter which folder you run them from.
CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


def test_same_seed_gives_same_data():
    """Running twice with the same seed must give identical results."""
    cfg = load_config(CONFIG)
    a, _ = run(cfg)
    b, _ = run(cfg)
    assert a["hourly_customers"].equals(b["hourly_customers"])


def test_different_seed_gives_different_data():
    """A different seed should produce different random data."""
    cfg = load_config(CONFIG)
    a, _ = run(cfg)
    b, _ = run(cfg.model_copy(update={"seed": cfg.seed + 1}))
    assert not a["hourly_customers"].equals(b["hourly_customers"])


def test_customers_match_true_rates_on_average():
    """
    Over two years, the total actual customers should be very close
    (within 1%) to the total expected customers. If not, the random
    draws are biased somewhere.
    """
    raw, truth = run(load_config(CONFIG))
    observed = raw["hourly_customers"]["customers"].sum()
    expected = truth.tables["hourly_rates"]["true_rate"].sum()
    assert abs(observed / expected - 1) < 0.01


def test_closed_hours_have_no_rows():
    """The output should only contain hours when the cafe is open."""
    cfg = load_config(CONFIG)
    raw, _ = run(cfg)
    assert set(raw["hourly_customers"]["hour"].unique()) == set(cfg.open_hours)