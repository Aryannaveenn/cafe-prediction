"""
The main simulation runner. It calls each module in order and collects
the results.

Right now the steps are:
  1. build the calendar
  2. work out expected customers per hour
  3. draw the actual customer counts

Later steps (menu, weather, staffing, inventory) will be added here.
"""
import polars as pl

from cafesim.config import SimConfig
from cafesim.context.calendar import build_calendar
from cafesim.demand.arrivals import draw_arrivals
from cafesim.demand.rate import build_hourly_rates
from cafesim.rng import RNGStreams
from cafesim.truth import TruthLog


def run(cfg: SimConfig) -> tuple[dict[str, pl.DataFrame], TruthLog]:
    """
    Run the whole simulation.

    Returns:
      raw   - tables a real cafe could observe (the model may use these)
      truth - hidden true values (only for scoring, never for training)
    """
    # One set of random streams for the whole run, all from the master seed.
    rngs = RNGStreams(cfg.seed)

    # Somewhere to collect the hidden true values as we go.
    truth = TruthLog()

    # Step 1: the calendar (dates, weekdays, holidays).
    calendar = build_calendar(cfg)

    # Step 2: expected customers per hour. Uses the "demand" random stream
    # for the daily shocks.
    hourly_rates, daily_factors = build_hourly_rates(cfg, calendar, rngs.get("demand"))

    # These are the true values behind the data, so they go in the truth log.
    truth.record("hourly_rates", hourly_rates)
    truth.record("daily_factors", daily_factors)

    # Step 3: actual customer counts. Uses its own "arrivals" random stream.
    arrivals = draw_arrivals(hourly_rates, rngs.get("arrivals"))

    # Everything here is something a real cafe would know.
    raw = {
        "calendar": calendar,  # dates and holidays are public information
        "hourly_customers": arrivals,
    }
    return raw, truth