"""Measure held-out sensor outage accuracy for the two PEMS-BAY models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import encode, training_data
from forecast_service import fill_missing
from model import HORIZON, build_model, forecast

CASES = ("clean", "15m_zero", "15m_carry", "60m_zero", "60m_carry")
SENSOR_OUTAGES = 33


@torch.inference_mode()
def score(model, speeds, clock, origins, config):
    rng = torch.Generator(device=speeds.device).manual_seed(271828)
    offsets = torch.arange(HORIZON, device=speeds.device)
    error = torch.zeros((len(CASES), 2), dtype=torch.float64, device=speeds.device)
    observations = torch.zeros(2, dtype=torch.int64, device=speeds.device)
    for indices in origins.split(64):
        rows = indices[:, None] + offsets
        values = speeds[rows - HORIZON]
        times = clock[rows - HORIZON]
        actual = speeds[rows]
        valid = actual > 0
        drop = (
            torch.rand(
                (len(indices), values.shape[-1]), generator=rng, device=speeds.device
            )
            .topk(SENSOR_OUTAGES, dim=1)
            .indices
        )
        affected = torch.zeros_like(values[:, 0], dtype=torch.bool)
        affected.scatter_(1, drop, True)
        relevant = valid & affected[:, None]
        observations += torch.stack((valid.sum(), relevant.sum()))
        for case_index, case in enumerate(CASES):
            current = values
            if case != "clean":
                window, policy = case.split("_")
                num_rows = 3 if window == "15m" else HORIZON
                current = values.clone()
                current[:, -num_rows:].masked_fill_(affected[:, None], 0.0)
                current = fill_missing(current, policy, config["mean_mph"])
            y = forecast(model, encode(current, times, config), config)
            difference = (y - actual).abs()
            error[case_index, 0] += difference.masked_select(valid).sum(
                dtype=torch.float64
            )
            error[case_index, 1] += difference.masked_select(relevant).sum(
                dtype=torch.float64
            )
    result = (error / observations).tolist()
    return {
        "windows": len(origins),
        "target_readings": int(observations[0]),
        "affected_target_readings": int(observations[1]),
        "cases": {
            name: {"all_mae_mph": all_sensors, "affected_mae_mph": affected_sensors}
            for name, (all_sensors, affected_sensors) in zip(CASES, result, strict=True)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("locations", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    speeds, clock, (_, validation, test), config = training_data(
        args.data, args.locations, 512, "cuda", 325, "America/Los_Angeles"
    )
    output = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "sensor_outages": SENSOR_OUTAGES,
        "mask_seed": 271828,
        "dataset_sha256": config["dataset_sha256"],
        "results": {},
    }
    for seed in (17, 29):
        for variant in ("baseline_coords", "cst"):
            tag = f"{variant}-s{seed}"
            arm = args.runs / tag
            metrics = json.loads((arm / "metrics.json").read_text())
            assert metrics["config"] == config
            model = build_model(variant, config).cuda().eval()
            model.load_state_dict(
                torch.load(arm / "best.pt", map_location="cpu", weights_only=True)
            )
            with torch.inference_mode():
                model.predefined_A = model.gc(model.idx)
                model.buildA_true = False
            validation_score = score(model, speeds, clock, validation, config)
            print(tag, "validation", json.dumps(validation_score), flush=True)
            test_score = score(model, speeds, clock, test, config)
            print(tag, "test", json.dumps(test_score), flush=True)
            assert (
                abs(
                    test_score["cases"]["clean"]["all_mae_mph"]
                    - metrics["test"]["mae_all"]
                )
                < 1e-5
            )
            output["results"][tag] = {
                "validation": validation_score,
                "test": test_score,
            }
            args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
            del model


if __name__ == "__main__":
    main()
