"""
Loads the simulation settings from a YAML file and checks they make sense.

Pydantic does the checking for us: if a setting is missing, the wrong type,
or out of range (like a negative number of days), it raises a clear error
before the simulation even starts.
"""
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class SimConfig(BaseModel):
    """
    Every setting the simulator understands.

    Each line is "setting_name: type = default". Field(...) adds rules,
    e.g. gt=0 means "must be greater than 0", ge=0 means "0 or more".
    """

    # --- Basic run settings ---
    name: str
    seed: int = 42
    start_date: date
    days: int = Field(gt=0)  # must simulate at least one day

    # --- Location (for public holidays) ---
    country: str = "AU"
    subdivision: str | None = None  # state/region, optional

    # --- Demand shape ---
    base_hourly_rate: float = Field(gt=0)
    hourly_profile: dict[int, float]  # hour -> relative busyness
    weekday_multipliers: list[float] = Field(min_length=7, max_length=7)  # exactly 7 days
    holiday_multiplier: float = Field(default=1.0, ge=0)

    # --- Longer-term patterns ---
    seasonal_amplitude: float = Field(default=0.0, ge=0, lt=1)
    seasonal_peak_day: int = Field(default=1, ge=1, le=366)
    trend_per_year: float = 0.0

    # --- Randomness ---
    daily_noise_sd: float = Field(default=0.0, ge=0)

    @field_validator("hourly_profile")
    @classmethod
    def check_profile(cls, profile: dict[int, float]) -> dict[int, float]:
        """Extra checks on the hourly profile that Field() can't express."""
        # The cafe must be open at least one hour a day.
        if not profile:
            raise ValueError("hourly_profile needs at least one open hour")

        for hour, value in profile.items():
            # Hours must be real clock hours.
            if not 0 <= hour <= 23:
                raise ValueError(f"hour {hour} must be between 0 and 23")
            # Busyness can't be negative.
            if value < 0:
                raise ValueError(f"hour {hour} has a negative value")

        # Return the profile sorted by hour, so 6, 7, 8... always comes out in order.
        return dict(sorted(profile.items()))

    @property
    def open_hours(self) -> list[int]:
        """The list of hours the cafe is open, e.g. [6, 7, 8, ..., 15]."""
        return list(self.hourly_profile)


def load_config(path: str | Path) -> SimConfig:
    """
    Read a YAML config file and turn it into a SimConfig.

    If the file contains `extends: some_other_file.yaml`, that other file is
    loaded first, and this file's settings override it. This lets scenario
    files list only what's different from the base.
    """
    path = Path(path)

    # Read the YAML into a plain Python dictionary.
    # `or {}` handles the case of an empty file.
    raw = yaml.safe_load(path.read_text()) or {}

    # Check whether this file builds on a parent file.
    parent = raw.pop("extends", None)
    if parent:
        # The parent path is relative to this file's folder.
        base = yaml.safe_load((path.parent / parent).read_text()) or {}
        base.pop("extends", None)  # we only support one level of inheritance

        # Start from the parent's settings, then overwrite with this file's.
        base.update(raw)
        raw = base

    # Hand the dictionary to Pydantic, which validates every setting.
    return SimConfig(**raw)