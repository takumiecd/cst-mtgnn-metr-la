"""Traffic speed loading, chronological target split and normalized model inputs."""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from model import HORIZON, NODES


def location_chart(path: Path, sensor_ids: list[str]) -> list[list[float]]:
    with path.open(newline="") as handle:
        entries = {
            row["sensor_id"]: (float(row["latitude"]), float(row["longitude"]))
            for row in csv.DictReader(handle)
        }
    coords = np.array([entries[sensor] for sensor in sensor_ids])
    latitude = coords[:, 0].mean()
    coords -= coords.mean(axis=0)
    coords *= (111.2, 111.2 * np.cos(np.deg2rad(latitude)))
    return coords.tolist()


def read_series(path: Path, expected_nodes: int = NODES, timezone: str | None = None):
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        timestamps, rows = [], []
        for row in reader:
            timestamps.append(row[0])
            rows.append(row[1:])
    sensor_ids = header[1:]
    speeds = np.asarray(rows, dtype=np.float32)
    if (
        speeds.ndim != 2
        or speeds.shape[1] != expected_nodes
        or len(sensor_ids) != expected_nodes
        or len(set(sensor_ids)) != expected_nodes
    ):
        raise ValueError(f"Expected {expected_nodes} unique sensor columns")
    if not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError("Expected finite, nonnegative traffic speeds")
    times = np.asarray(timestamps, dtype="datetime64[m]")
    cadence = times
    if timezone is not None:
        zone = ZoneInfo(timezone)
        cadence = np.array(
            [
                datetime.fromisoformat(t).replace(tzinfo=zone).timestamp()
                for t in timestamps
            ],
            dtype=np.int64,
        ).astype("datetime64[s]")
    if not (np.diff(cadence) == np.timedelta64(5, "m")).all():
        raise ValueError("Expected contiguous five-minute timestamps")
    return sensor_ids, speeds, times


def time_of_day(times):
    return (times - times.astype("datetime64[D]")) / np.timedelta64(1, "D")


def split_origins(count: int):
    train_end, val_end = round(0.7 * count), round(0.8 * count)
    return (
        np.arange(HORIZON, train_end - HORIZON + 1),
        np.arange(train_end, val_end - HORIZON + 1),
        np.arange(val_end, count - HORIZON + 1),
    )


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def training_data(
    path: Path,
    locations: Path,
    atoms: int,
    device: str,
    expected_nodes: int = NODES,
    timezone: str | None = None,
):
    sensor_ids, speeds, times = read_series(path, expected_nodes, timezone)
    clock = time_of_day(times)
    origins = split_origins(len(speeds))
    offsets = np.arange(-HORIZON, 0)
    train_input = speeds[origins[0][:, None] + offsets]
    config = {
        "atoms": atoms,
        "sensor_ids": sensor_ids,
        "coordinates_km": location_chart(locations, sensor_ids),
        "mean_mph": float(train_input.mean(dtype=np.float64)),
        "std_mph": float(train_input.std(dtype=np.float64)),
        "dataset_sha256": sha256_file(path),
        "split_counts": [len(index) for index in origins],
    }
    speeds = torch.as_tensor(speeds, device=device)
    clock = torch.tensor(clock, device=device, dtype=torch.float32)
    ranges = tuple(torch.as_tensor(index, device=device) for index in origins)
    return speeds, clock, ranges, config


def encode(speeds: torch.Tensor, clock: torch.Tensor, config: dict) -> torch.Tensor:
    normalized = (speeds - config["mean_mph"]) / config["std_mph"]
    time = clock.unsqueeze(-1).expand_as(normalized)
    return torch.stack((normalized, time), dim=1).transpose(2, 3).contiguous()


def get_batch(speeds, clock, origins, config):
    steps = torch.arange(HORIZON, device=speeds.device)
    inputs = origins[:, None] + steps - HORIZON
    targets = origins[:, None] + steps
    return encode(speeds[inputs], clock[inputs], config), speeds[targets]
