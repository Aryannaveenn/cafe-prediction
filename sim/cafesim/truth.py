"""
Stores the hidden "ground truth" behind the simulated data.

The simulator knows things a real cafe never could, like the exact expected
number of customers each hour. We save these separately (to data/truth/)
so that later we can check how close the prediction model got to reality.

IMPORTANT: the prediction model must never read these files during training,
or it would be cheating.
"""
import polars as pl


class TruthLog:
    def __init__(self):
        # A dictionary of table name -> table of true values.
        self.tables: dict[str, pl.DataFrame] = {}

    def record(self, name: str, df: pl.DataFrame) -> None:
        """Save a table of true values under a name, e.g. "hourly_rates"."""
        self.tables[name] = df