"""
Builds the calendar: one row for every simulated day.

Each row says what day of the week it is, what day of the year it is,
and whether it's a public holiday. Other modules use this to decide
how busy each day should be.
"""
from datetime import timedelta

import holidays
import polars as pl

from cafesim.config import SimConfig


def build_calendar(cfg: SimConfig) -> pl.DataFrame:
    # Make a list of every date: start_date, the day after, and so on.
    dates = [cfg.start_date + timedelta(days=i) for i in range(cfg.days)]

    # Look up the public holidays for every year our dates cover.
    years = sorted({d.year for d in dates})
    public = holidays.country_holidays(cfg.country, subdiv=cfg.subdivision, years=years)

    # Build the table, one column at a time.
    return pl.DataFrame(
        {
            "date": dates,
            # Day of the week as a number: 0 = Monday ... 6 = Sunday.
            "weekday": [d.weekday() for d in dates],
            # Day of the year: 1 = Jan 1st ... 365 (or 366) = Dec 31st.
            "day_of_year": [d.timetuple().tm_yday for d in dates],
            # True if this date is a public holiday.
            "is_holiday": [d in public for d in dates],
            # The holiday's name, or empty (null) on normal days.
            "holiday_name": [public.get(d) for d in dates],
        },
        # Tell Polars this column holds text, even though most values are empty.
        schema_overrides={"holiday_name": pl.String},
    )