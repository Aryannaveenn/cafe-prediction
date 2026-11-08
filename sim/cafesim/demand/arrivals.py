"""
Turns EXPECTED customers into ACTUAL customers.

If we expect 44 customers at 8am, the real number won't be exactly 44.
Some days it's 39, some days 51. The Poisson distribution is the standard
way to model "random counts around an average": it's what you get when
many people independently decide whether to walk in.
"""
import numpy as np
import polars as pl


def draw_arrivals(hourly_rates: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    # Draw one random count for every (date, hour), using that hour's
    # expected rate as the average.
    counts = rng.poisson(hourly_rates["true_rate"].to_numpy())

    # Return only what a real cafe could observe: the date, the hour, and
    # how many customers came. The true rate is deliberately left out.
    return hourly_rates.select("date", "hour").with_columns(
        pl.Series("customers", counts, dtype=pl.Int64)
    )