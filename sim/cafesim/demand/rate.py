"""
Works out how many customers we EXPECT in every open hour.

The expected number (the "rate") is the base rate multiplied by several
factors, each describing one real-world effect:

    rate = base × hour × weekday × holiday × season × trend × random_shock

Example: base 40, at 8am (×1.0) on a Friday (×1.10), not a holiday (×1.0),
average season (×1.0), no growth yet (×1.0), normal luck (×1.0)
-> we expect 40 × 1.10 = 44 customers between 8am and 9am.

This is only the EXPECTED number. The actual count is drawn randomly
around it in arrivals.py.
"""
import numpy as np
import polars as pl

from cafesim.config import SimConfig


def build_hourly_rates(
    cfg: SimConfig, calendar: pl.DataFrame, rng: np.random.Generator
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Returns two tables:
      - hourly: the expected customers for every (date, hour)
      - daily_factors: each day's individual factors, saved as ground truth
    """
    n_days = calendar.height  # number of rows = number of days

    # Pull the calendar columns out as NumPy arrays so we can do maths on
    # all days at once, instead of looping one day at a time.
    weekday = calendar["weekday"].to_numpy()
    day_of_year = calendar["day_of_year"].to_numpy()
    is_holiday = calendar["is_holiday"].to_numpy()

    # ---------- Day-level factors (one number per day) ----------

    # Weekday effect: look up each day's multiplier from the config list.
    # e.g. weekday 4 (Friday) picks weekday_multipliers[4] = 1.10.
    weekday_f = np.array(cfg.weekday_multipliers)[weekday]

    # Holiday effect: use the holiday multiplier on holidays, 1.0 otherwise.
    holiday_f = np.where(is_holiday, cfg.holiday_multiplier, 1.0)

    # Seasonal effect: a smooth wave over the year using cosine.
    # It equals (1 + amplitude) on the peak day, (1 - amplitude) half a
    # year later, and moves smoothly in between.
    season_f = 1 + cfg.seasonal_amplitude * np.cos(
        2 * np.pi * (day_of_year - cfg.seasonal_peak_day) / 365.25
    )

    # Growth trend: compound growth, like interest on savings.
    # Day 0 = 1.0, one year later = 1 + trend_per_year, and so on.
    trend_f = (1 + cfg.trend_per_year) ** (np.arange(n_days) / 365.25)

    # Random daily shock: unexplained good or bad days.
    # We use a lognormal distribution because it's always positive and
    # skews slightly upward (occasional surprise busy days). The first
    # argument (-0.5 × sd²) is a correction that makes its average exactly 1,
    # so the shocks don't secretly raise or lower overall demand.
    sd = cfg.daily_noise_sd
    if sd > 0:
        shock = rng.lognormal(-0.5 * sd**2, sd, n_days)
    else:
        shock = np.ones(n_days)  # no randomness requested

    # Multiply all the day-level effects together.
    day_f = weekday_f * holiday_f * season_f * trend_f * shock

    # ---------- Combine with the hour-of-day pattern ----------

    hours = cfg.open_hours  # e.g. [6, 7, ..., 15]
    hour_f = np.array([cfg.hourly_profile[h] for h in hours])

    # np.outer makes a grid: one row per day, one column per hour.
    # Each cell = that day's factor × that hour's factor.
    # Multiplying by the base rate gives expected customers per cell.
    rates = cfg.base_hourly_rate * np.outer(day_f, hour_f)

    # ---------- Turn the grid into a long table ----------
    # One row per (date, hour), which is easier to save and analyse.

    dates = calendar["date"].to_list()
    hourly = pl.DataFrame(
        {
            # Each date repeated once for every open hour.
            "date": [d for d in dates for _ in hours],
            # The list of hours repeated once for every day.
            "hour": hours * n_days,
            # Flatten the grid row by row so it lines up with the columns above.
            "true_rate": rates.ravel(),
        }
    )

    # Save each day's individual factors too. These are ground truth: later
    # we can check whether the model figured out, say, the holiday effect.
    daily_factors = pl.DataFrame(
        {
            "date": dates,
            "weekday_factor": weekday_f,
            "holiday_factor": holiday_f,
            "season_factor": season_f,
            "trend_factor": trend_f,
            "random_shock": shock,
        }
    )

    return hourly, daily_factors