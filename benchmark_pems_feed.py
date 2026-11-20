"""Replay PEMS-BAY forecasts under causal sensor outages and feed latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from data import encode, training_data
from forecast_service import fill_missing
from model import HORIZON, build_model, forecast

EVENT_PERIOD = 72  # six hours in five-minute slots
NEIGHBORS = 33
MASK_SEED = 271828
GROUPS = ("all", "affected_windows", "affected_sensors", "affected_slow")
SEEDS = (17, 29, 43)
VARIANTS = ("baseline_coords", "cst")
MINUTES = (15, 30, 60)


@dataclass(frozen=True)
class Condition:
    name: str
    outage_slots: int = 0
    regional_latency_slots: int = 0
    network_latency_slots: int = 0
    carry: bool = True


CASES = (
    Condition("clean_raw", carry=False),
    Condition("clean_service"),
    Condition("regional_outage_30m", outage_slots=6),
    Condition("regional_outage_120m", outage_slots=24),
    Condition("regional_delay_15m", regional_latency_slots=3),
    Condition("network_delay_5m", network_latency_slots=1),
    Condition("network_delay_15m", network_latency_slots=3),
    Condition("network_delay_30m", network_latency_slots=6),
    Condition(
        "regional_outage_120m_network_delay_15m",
        outage_slots=24,
        network_latency_slots=3,
    ),
)


def event_timeline(coordinates, start, stop):
    """Rotate one nearby group and one outage start per six-hour interval."""
    distances = np.square(coordinates[:, None] - coordinates[None, :]).sum(axis=-1)
    nearby = np.argsort(distances, axis=1, kind="stable")[:, :NEIGHBORS]
    region = np.zeros((stop - start, len(coordinates)), dtype=bool)
    age = np.full(stop - start, -1, dtype=np.int16)
    random = np.random.default_rng(MASK_SEED)
    for period_start in range(0, len(age), EVENT_PERIOD):
        end = min(period_start + EVENT_PERIOD, len(age))
        anchor = random.integers(len(coordinates))
        event = random.integers(EVENT_PERIOD - 24 + 1)
        region[period_start:end, nearby[anchor]] = True
        event_start = period_start + event
        if event_start < end:
            age[event_start:end] = np.arange(end - event_start)
    return region, age


def withheld_at_issue(case, ticks, origins, region, age):
    """A sample recorded at t and delayed by d is available for origin o if t+d < o."""
    outage = region & ((age >= 0) & (age < case.outage_slots)).unsqueeze(-1)
    delay = case.network_latency_slots
    if case.regional_latency_slots:
        delay = delay + region.to(torch.int16) * case.regional_latency_slots
    late = delay >= (origins[:, None] - ticks).unsqueeze(-1)
    return outage | late


def protocol(config, test, device):
    first, stop = int(test[0]) - HORIZON, int(test[-1])
    coordinates = np.asarray(config["coordinates_km"])
    region, age = event_timeline(coordinates, first, stop)
    digest = hashlib.sha256(region.tobytes() + age.tobytes()).hexdigest()
    return (
        first,
        torch.as_tensor(region, device=device),
        torch.as_tensor(age, device=device),
        digest,
    )


def metrics(errors, counts):
    def mae(a, n):
        return a / n if n else None

    errors, counts = errors.tolist(), counts.tolist()
    return {
        name: {
            "mae_mph": mae(sum(row), sum(denom)),
            "observations": sum(denom),
            "horizons_mph": {
                str(minute): mae(row[minute // 5 - 1], denom[minute // 5 - 1])
                for minute in MINUTES
            },
        }
        for name, row, denom in zip(GROUPS, errors, counts, strict=True)
    }


@torch.inference_mode()
def score(model, speeds, clock, origins, config, timeline, batch_size, cases):
    base, region_timeline, ages, _ = timeline
    device = speeds.device
    offsets = torch.arange(HORIZON, device=device)
    sums = torch.zeros(
        (len(cases), len(GROUPS), HORIZON), device=device, dtype=torch.float64
    )
    counts = torch.zeros_like(sums, dtype=torch.int64)
    impacted = torch.zeros(len(cases), device=device, dtype=torch.int64)
    for batch in origins.split(batch_size):
        target_ticks = batch[:, None] + offsets
        ticks = target_ticks - HORIZON
        actual, observed, times = speeds[target_ticks], speeds[ticks], clock[ticks]
        valid = actual > 0
        slow = actual < 45
        region, age = region_timeline[ticks - base], ages[ticks - base]
        for idx, case in enumerate(cases):
            withheld = withheld_at_issue(case, ticks, batch, region, age)
            sensors = (withheld & (observed > 0)).any(dim=1)
            windows = sensors.any(dim=-1)
            impacted[idx] += windows.sum()
            data = observed.masked_fill(withheld, 0)
            if case.carry:
                data = fill_missing(data, "carry", config["mean_mph"])
            prediction = forecast(model, encode(data, times, config), config)
            error = (prediction - actual).abs()
            affected = valid & sensors[:, None]
            groups = (valid, valid & windows[:, None, None], affected, affected & slow)
            for group, mask in enumerate(groups):
                sums[idx, group] += error.masked_fill(~mask, 0).sum(
                    (0, 2), dtype=torch.float64
                )
                counts[idx, group] += mask.sum((0, 2))
    assert torch.isfinite(sums).all()
    return {
        case.name: {
            "affected_window_count": int(impacted[idx]),
            **metrics(sums[idx], counts[idx]),
        }
        for idx, case in enumerate(cases)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("locations", type=Path)
    parser.add_argument(
        "runs", type=Path, nargs=2, help="Run directories for seeds 17/29 and 43"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.get_device_name() == "NVIDIA GeForce RTX 4090"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    speeds, clock, (_, _, test), config = training_data(
        args.data, args.locations, 512, "cuda", 325, "America/Los_Angeles"
    )
    timeline = protocol(config, test, speeds.device)
    seeds = (17,) if args.smoke else SEEDS
    origins = test[:128] if args.smoke else test
    report = {
        "dataset_sha256": config["dataset_sha256"],
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "seeds": seeds,
        "windows": len(origins),
        "conditions": [asdict(c) for c in CASES],
        "timeline": {
            "seed": MASK_SEED,
            "period_slots": EVENT_PERIOD,
            "neighbor_sensors": NEIGHBORS,
            "sha256": timeline[-1],
        },
        "results": {},
    }
    for seed in seeds:
        run = args.runs[0 if seed in (17, 29) else 1]
        for variant in VARIANTS:
            key = f"{variant}-s{seed}"
            directory = run / "results" / key
            run_metrics = json.loads((directory / "metrics.json").read_text())
            assert run_metrics["config"] == config and run_metrics["epochs"] == 100
            assert run_metrics["seed"] == seed and run_metrics["variant"] == variant
            model = build_model(variant, config).cuda().eval()
            model.load_state_dict(
                torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            )
            with torch.inference_mode():
                model.predefined_A = model.gc(model.idx)
                model.buildA_true = False
            start = time.perf_counter()
            scores = score(
                model,
                speeds,
                clock,
                origins,
                config,
                timeline,
                run_metrics["batch_size"],
                CASES,
            )
            if not args.smoke:
                clean = scores["clean_raw"]["all"]["mae_mph"]
                assert abs(clean - run_metrics["test"]["mae_all"]) < 1e-5, (key, clean)
            report["results"][key] = {
                "selected_epoch": run_metrics["best_epoch"],
                "seconds": time.perf_counter() - start,
                "cases": scores,
            }
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            tmp.replace(args.output)
            print(
                key,
                {name: round(r["all"]["mae_mph"], 4) for name, r in scores.items()},
                flush=True,
            )
            del model


if __name__ == "__main__":
    main()
