"""Compare live and precomputed adjacency on repeated MTGNN inference."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torchcst.hub import load_safetensors

from data import encode, read_series, sha256_file, split_origins, time_of_day
from model import HORIZON, build_model, forecast


def benchmark(model, inputs, config, repeats):
    for _ in range(5):
        forecast(model, inputs, config)
    times_ms = []
    for _ in range(repeats):
        start = perf_counter()
        forecast(model, inputs, config)
        times_ms.append((perf_counter() - start) * 1000)
    return median(times_ms)


@torch.inference_mode()
def compare(model, inputs, config, repeats):
    model.eval()
    graph = model.gc(model.idx)
    result = {"adjacency_bytes": graph.nbytes, "measurements": []}
    try:
        for batch in (1, 8):
            fixture = inputs[:batch]
            model.buildA_true = True
            reference = forecast(model, fixture, config)
            live_ms = benchmark(model, fixture, config, repeats)
            model.predefined_A = graph
            model.buildA_true = False
            cached = forecast(model, fixture, config)
            cached_ms = benchmark(model, fixture, config, repeats)
            torch.testing.assert_close(cached, reference, atol=1e-5, rtol=1e-6)
            result["measurements"].append(
                {
                    "batch": batch,
                    "live_median_ms": live_ms,
                    "precomputed_median_ms": cached_ms,
                    "max_absolute_prediction_error": float(
                        (cached - reference).abs().max()
                    ),
                }
            )
    finally:
        model.predefined_A = None
        model.buildA_true = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    repeats = 40
    config_path = args.weights / "config.json"
    config = json.loads(config_path.read_text())
    assert sha256_file(args.data) == config["dataset_sha256"]
    sensors, speeds, times = read_series(args.data)
    assert sensors == config["sensor_ids"]
    origins = split_origins(len(speeds))[-1][:8]
    indices = origins[:, None] + torch.arange(-HORIZON, 0).numpy()
    x = encode(
        torch.from_numpy(speeds[indices]),
        torch.tensor(time_of_day(times[indices]), dtype=torch.float32),
        config,
    )
    result = {
        "device": "cpu",
        "torch": str(torch.__version__),
        "threads": torch.get_num_threads(),
        "repeats": repeats,
        "config_sha256": sha256_file(config_path),
        "arms": {},
    }
    for variant in ("baseline", "cst"):
        path = args.weights / f"{variant}.safetensors"
        model = load_safetensors(
            path, model_factory=partial(build_model, variant, config), device="cpu"
        )
        result["arms"][variant] = compare(model, x, config, repeats)
        result["arms"][variant]["weight_sha256"] = sha256_file(path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
