"""Serve a loaded traffic model over a local HTTP API."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torchcst.hub import load_safetensors

from data import encode
from model import HORIZON, build_model, forecast

INTERVAL = timedelta(minutes=5)
MISSING_POLICIES = ("zero", "carry")
MAX_REQUEST_BYTES = 128 * 1024
MAX_FLOAT32 = torch.finfo(torch.float32).max


def fill_missing(speeds: torch.Tensor, policy: str, default_mph: float) -> torch.Tensor:
    """Use the latest positive reading, with the training mean for leading gaps."""
    if policy == "zero":
        return speeds
    if policy != "carry":
        raise ValueError("Unknown missing-value policy")
    slot = torch.arange(1, speeds.size(1) + 1, device=speeds.device)[None, :, None]
    observed = torch.where(speeds > 0, slot, 0)
    last = observed.cummax(dim=1).values
    recovered = speeds.gather(1, (last - 1).clamp_min(0))
    return torch.where(last > 0, recovered, default_mph)


class TrafficForecaster:
    def __init__(self, model, config, variant, missing_policy):
        self.model = model.eval()
        self.config = config
        self.variant = variant
        self.missing_policy = missing_policy
        self.sensors = config["sensor_ids"]
        self.columns = {sensor: index for index, sensor in enumerate(self.sensors)}
        self.zone = ZoneInfo(config.get("timezone", "America/Los_Angeles"))
        self.device = next(model.parameters()).device
        with torch.inference_mode():
            self.model.predefined_A = self.model.gc(self.model.idx)
            self.model.buildA_true = False

    def parse_times(self, raw_times):
        if not isinstance(raw_times, list) or len(raw_times) != HORIZON:
            raise ValueError("timestamps must contain 12 five-minute readings")
        times = []
        for raw in raw_times:
            if not isinstance(raw, str):
                raise TypeError("timestamps must be ISO 8601 strings")
            t = datetime.fromisoformat(raw)
            if t.utcoffset() is None:
                raise ValueError("timestamps must include a UTC offset")
            times.append(t.astimezone(UTC))
        if any(t.minute % 5 or t.second or t.microsecond for t in times) or any(
            b - a != INTERVAL for a, b in pairwise(times)
        ):
            raise ValueError(
                "timestamps must be consecutive and on a five-minute interval"
            )
        return times

    def parse_speeds(self, given, readings):
        if (
            not isinstance(given, list)
            or not given
            or any(not isinstance(s, str) or s not in self.columns for s in given)
            or len(set(given)) != len(given)
        ):
            raise ValueError("sensor_ids must be unique known IDs")
        if not isinstance(readings, list) or len(readings) != HORIZON:
            raise ValueError("speeds_mph must contain one row per timestamp")
        speeds = [[0.0] * len(self.sensors) for _ in range(HORIZON)]
        by_request = [self.columns[sensor] for sensor in given]
        for out, row in zip(speeds, readings, strict=True):
            if not isinstance(row, list) or len(row) != len(given):
                raise ValueError("each speed row must have one value per sensor_id")
            for column, raw in zip(by_request, row, strict=True):
                if raw is None:
                    continue
                if (
                    type(raw) not in (int, float)
                    or not 0 <= raw <= MAX_FLOAT32
                    or not math.isfinite(raw)
                ):
                    raise ValueError(
                        "speed readings must be finite nonnegative numbers or null"
                    )
                out[column] = raw
        tensor = torch.tensor(speeds, dtype=torch.float32, device=self.device)
        if not (tensor > 0).any():
            raise ValueError("at least one speed reading must be available")
        return tensor[None]

    @torch.inference_mode()
    def predict(self, payload):
        if not isinstance(payload, dict):
            raise TypeError("request must be a JSON object")
        times = self.parse_times(payload.get("timestamps"))
        speeds = self.parse_speeds(payload.get("sensor_ids"), payload.get("speeds_mph"))
        local = [t.astimezone(self.zone) for t in times]
        clock = torch.tensor(
            [[(t.hour * 60 + t.minute) / 1440 for t in local]],
            dtype=torch.float32,
            device=self.device,
        )
        fixed = fill_missing(speeds, self.missing_policy, self.config["mean_mph"])
        output = forecast(self.model, encode(fixed, clock, self.config), self.config)
        if not torch.isfinite(output).all():
            raise ValueError("forecast produced a non-finite value")
        future = [
            (times[-1] + step * INTERVAL)
            .astimezone(self.zone)
            .isoformat(timespec="minutes")
            for step in range(1, HORIZON + 1)
        ]
        return {
            "variant": self.variant,
            "missing_policy": self.missing_policy,
            "missing_readings": int((speeds == 0).sum()),
            "sensors_without_readings": int((speeds == 0).all(dim=1).sum()),
            "sensor_ids": self.sensors,
            "timestamps": future,
            "speeds_mph": output[0].cpu().tolist(),
        }


def handler_for(service):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(
                    200, {"variant": service.variant, "sensors": len(service.sensors)}
                )
            else:
                self.send_json(404, {"error": "unknown route"})

        def do_POST(self):
            if self.path != "/forecast":
                self.send_json(404, {"error": "unknown route"})
                return
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.send_json(411, {"error": "valid Content-Length required"})
                return
            if not 0 < size <= MAX_REQUEST_BYTES:
                self.send_json(413, {"error": "invalid request size"})
                return
            try:
                request = json.loads(self.rfile.read(size))
                result = service.predict(request)
            except (ValueError, TypeError, OverflowError) as exc:
                self.send_json(400, {"error": str(exc)})
                return
            self.send_json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("baseline", "baseline_coords", "cst"), default="cst"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--missing-policy", choices=MISSING_POLICIES, default="carry")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    model = load_safetensors(
        args.weights,
        model_factory=partial(build_model, args.variant, config),
        device=args.device,
    )
    service = TrafficForecaster(model, config, args.variant, args.missing_policy)
    with HTTPServer(("127.0.0.1", args.port), handler_for(service)) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
