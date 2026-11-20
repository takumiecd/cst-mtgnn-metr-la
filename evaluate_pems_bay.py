"""Evaluate the paired PEMS-BAY models against persistence on held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import get_batch, training_data
from model import build_model, forecast

THRESHOLD_MPH = 45.0


@torch.inference_mode()
def evaluate(model, speeds, clock, origins, config, batch_size):
    all_count = torch.zeros(speeds.shape[1], dtype=torch.int64, device=speeds.device)
    all_error = all_count.double()
    slow_count = all_count.clone()
    slow_error = all_error.clone()
    for index in origins.split(batch_size):
        inputs, targets = get_batch(speeds, clock, index, config)
        if model is None:
            last_speed = speeds[index - 1]
            last_speed = torch.where(last_speed > 0, last_speed, config["mean_mph"])
            predictions = last_speed[:, None, :]
        else:
            predictions = forecast(model, inputs, config)
        if not torch.isfinite(predictions).all():
            raise ValueError("Nonfinite test predictions")
        error = (predictions - targets).abs()
        mask = targets > 0
        slow = mask & (targets < THRESHOLD_MPH)
        all_count += mask.sum((0, 1))
        all_error += (error * mask).sum((0, 1), dtype=torch.float64)
        slow_count += slow.sum((0, 1))
        slow_error += (error * slow).sum((0, 1), dtype=torch.float64)
    return {
        "mae_all_mph": float(all_error.sum() / all_count.sum()),
        "mae_below_45_mph": float(slow_error.sum() / slow_count.sum()),
        "all_count": int(all_count.sum()),
        "below_45_count": int(slow_count.sum()),
        "sensor_mae_mph": [
            float(n / d) if d else None
            for n, d in zip(all_error.tolist(), all_count.tolist(), strict=True)
        ],
    }


@torch.inference_mode()
def graph_summary(model, config):
    adjacent = model.gc(model.idx) > 0
    coordinates = torch.tensor(config["coordinates_km"], device=adjacent.device)
    output, source = adjacent.nonzero(as_tuple=True)
    distance = (coordinates[output] - coordinates[source]).norm(dim=1)
    return {
        "positive_edges": distance.numel(),
        "nodes_without_incoming_edges": int((~adjacent.any(1)).sum()),
        "nodes_without_outgoing_edges": int((~adjacent.any(0)).sum()),
        "median_edge_km": float(distance.median()) if distance.numel() else None,
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
    speeds, clock, (_, _, test), config = training_data(
        args.data, args.locations, 512, "cuda", 325, "America/Los_Angeles"
    )
    results = {"sensor_ids": config["sensor_ids"], "runs": {}}
    results["persistence"] = evaluate(None, speeds, clock, test, config, 64)
    for seed in (17, 29):
        for variant in ("baseline_coords", "cst"):
            tag = f"{variant}-s{seed}"
            run = args.runs / tag
            metrics = json.loads((run / "metrics.json").read_text())
            assert metrics["config"] == config
            model = build_model(variant, config)
            model.load_state_dict(
                torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            )
            model.cuda().eval()
            row = evaluate(model, speeds, clock, test, config, metrics["batch_size"])
            assert abs(row["mae_all_mph"] - metrics["test"]["mae_all"]) < 1e-5
            row["graph"] = graph_summary(model, config)
            results["runs"][tag] = row
            del model
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
