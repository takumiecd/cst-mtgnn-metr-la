"""Forecast an hour of traffic speeds for the sensors in a checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torchcst.hub import from_pretrained, load_safetensors

from data import encode, read_series, sha256_file, split_origins, time_of_day
from model import HORIZON, build_model, forecast

HUB_REPO = "takumiecd/cst-mtgnn-metr-la"


@torch.inference_mode()
def predict(model, config, sensor_ids, speeds, times, origin, sensor_id):
    expected = config["sensor_ids"]
    if set(sensor_ids) != set(expected):
        raise ValueError("The CSV must contain all sensor IDs in the checkpoint")
    if not HORIZON <= origin <= len(speeds):
        raise ValueError("A complete hour of input is required")
    column = expected.index(sensor_id)
    device = next(model.parameters()).device
    speed_order = [sensor_ids.index(sensor) for sensor in expected]
    speed_window = torch.as_tensor(
        speeds[origin - HORIZON : origin, speed_order], device=device
    ).unsqueeze(0)
    clock_window = torch.tensor(
        time_of_day(times[origin - HORIZON : origin]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    result = forecast(model, encode(speed_window, clock_window, config), config)
    rows = []
    start = times[origin - 1]
    zone_name = config.get("timezone")
    if zone_name:
        zone = ZoneInfo(zone_name)
        utc_start = (
            datetime.fromisoformat(str(start)).replace(tzinfo=zone).astimezone(UTC)
        )
    for step, value in enumerate(result[0, :, column].tolist()):
        minutes = 5 * (step + 1)
        if zone_name:
            time = (
                (utc_start + timedelta(minutes=minutes))
                .astimezone(zone)
                .isoformat(timespec="minutes")
            )
        else:
            time = str(start + np.timedelta64(minutes, "m"))
        row = {"time": time, "prediction_mph": value}
        if origin + step < len(speeds):
            observed = speeds[origin + step, speed_order[column]]
            row["observed_mph"] = float(observed) if observed > 0 else None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--variant", choices=("baseline", "baseline_coords", "cst"), default="cst"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sensor", help="METR-LA sensor ID, defaults to the first")
    parser.add_argument("--test-index", type=int, help="Zero-based held-out forecast")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--revision", help="Hugging Face commit or tag")
    args = parser.parse_args()
    if (args.config is None) != (args.weights is None):
        parser.error("Use --config and --weights together for local files")
    if args.variant == "baseline_coords" and args.weights is None:
        parser.error("The coordinate baseline requires local --config and --weights")

    path = args.config or hf_hub_download(
        HUB_REPO, "config.json", revision=args.revision, token=False
    )
    config = json.loads(Path(path).read_text())
    sensor_id = args.sensor or config["sensor_ids"][0]
    if sensor_id not in config["sensor_ids"]:
        parser.error("Unknown METR-LA sensor ID")
    sensor_ids, speeds, times = read_series(
        args.input, len(config["sensor_ids"]), config.get("timezone")
    )
    origin = len(speeds)
    if args.test_index is not None:
        test = split_origins(len(speeds))[-1]
        if (
            not 0 <= args.test_index < len(test)
            or sha256_file(args.input) != config["dataset_sha256"]
        ):
            parser.error(
                "--test-index requires the reference CSV and a valid test index"
            )
        origin = int(test[args.test_index])

    factory = partial(build_model, args.variant, config)
    if args.weights is None:
        model = from_pretrained(
            HUB_REPO,
            model_factory=factory,
            filename=f"{args.variant}.safetensors",
            device=args.device,
            revision=args.revision,
            token=False,
        )
    else:
        model = load_safetensors(
            args.weights, model_factory=factory, device=args.device
        )
    rows = predict(model, config, sensor_ids, speeds, times, origin, sensor_id)
    print(
        json.dumps(
            {"variant": args.variant, "sensor_id": sensor_id, "forecast": rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
