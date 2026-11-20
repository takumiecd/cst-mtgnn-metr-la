"""Tune a calendar-and-persistence traffic forecast on the validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import read_series, sha256_file, split_origins
from model import HORIZON

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
SLOTS_PER_DAY = 288
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7


def time_slot(times):
    days = times.astype("datetime64[D]")
    weekday = (days.astype(np.int64) + 3) % 7
    minute = (times - days).astype("timedelta64[m]").astype(np.int64)
    return weekday * SLOTS_PER_DAY + minute // 5


def calendar_profile(speeds, slots, train_end):
    history = speeds[:train_end].copy()
    history[history <= 0] = np.nan
    fallback = np.nanmedian(history, axis=0)
    if not np.isfinite(fallback).all():
        raise ValueError("A sensor has no available training readings")
    profile = np.stack(
        [
            np.nanmedian(history[slots[:train_end] == k], axis=0)
            for k in range(SLOTS_PER_WEEK)
        ]
    )
    return np.where(np.isfinite(profile), profile, fallback)


def summarize(speeds, profile, slots, validation, test):
    out = {"target_count": 0, "below_45_count": 0, "horizons": []}
    absolute = slow_absolute = 0.0
    for horizon in range(HORIZON):
        actual = speeds[validation + horizon]
        nominal = profile[slots[validation + horizon]]
        last = speeds[validation - 1]
        last_profile = profile[slots[validation - 1]]
        offset = np.where(last > 0, last - last_profile, 0.0)
        observed = actual > 0
        weight = min(
            WEIGHTS,
            key=lambda w: np.abs(nominal + w * offset - actual)[observed].mean(),
        )
        actual = speeds[test + horizon]
        nominal = profile[slots[test + horizon]]
        last = speeds[test - 1]
        offset = np.where(last > 0, last - profile[slots[test - 1]], 0.0)
        error = np.abs(nominal + weight * offset - actual)
        observed = actual > 0
        slow = observed & (actual < 45.0)
        errors = error[observed]
        absolute += errors.sum(dtype=np.float64)
        slow_absolute += error[slow].sum(dtype=np.float64)
        out["target_count"] += errors.size
        out["below_45_count"] += int(slow.sum())
        out["horizons"].append(
            {
                "minutes": 5 * (horizon + 1),
                "weight": weight,
                "mae_mph": float(errors.mean()),
            }
        )
    out["mae_all_mph"] = float(absolute / out["target_count"])
    out["mae_below_45_mph"] = float(slow_absolute / out["below_45_count"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    _, speeds, times = read_series(args.data, 325, "America/Los_Angeles")
    slots = time_slot(times)
    train, validation, test = split_origins(len(times))
    profile = calendar_profile(speeds, slots, train[-1] + HORIZON)
    result = summarize(speeds, profile, slots, validation, test)
    result["dataset_sha256"] = sha256_file(args.data)
    result["train_observations_end_exclusive"] = int(train[-1] + HORIZON)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["mae_all_mph"], result["mae_below_45_mph"])


if __name__ == "__main__":
    main()
