"""Train and evaluate the two MTGNN graph parameterizations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch import nn

from data import get_batch, training_data
from model import HORIZON, build_model, forecast
from torchcst import CSTAdamW


def shared_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not name.startswith("gc."):
            digest.update(name.encode())
            digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def step(model, optimizer, inputs, targets, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = forecast(model, inputs, config)
    mask = targets > 0
    loss = ((pred - targets).abs() * mask).sum() / mask.sum().clamp_min(1)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.detach()


def make_optimizer(model, args):
    options = {"lr": 0.001, "weight_decay": 0.0001, "foreach": True}
    if args.variant == "baseline":
        return torch.optim.AdamW(model.parameters(), **options)
    atoms = model.gc.linear.atoms.p
    groups = [
        {"params": [p for p in model.parameters() if p is not atoms]},
        {"params": [atoms], "weight_decay": 0.0, "lr": 0.0048},
    ]
    return CSTAdamW(
        groups, model=model, amplitude_weight_decay=options["weight_decay"], **options
    )


@torch.inference_mode()
def measure(model, speeds, clock, origins, config, batch_size):
    model.eval()
    counts = torch.zeros(HORIZON, dtype=torch.float64, device=speeds.device)
    absolute = counts.clone()
    squares = counts.clone()
    percentages = counts.clone()
    for index in origins.split(batch_size):
        inputs, targets = get_batch(speeds, clock, index, config)
        pred = forecast(model, inputs, config)
        mask = targets > 0
        error = (pred - targets).abs()
        counts += mask.sum((0, 2))
        absolute += (error * mask).sum((0, 2), dtype=torch.float64)
        squares += (error.square() * mask).sum((0, 2), dtype=torch.float64)
        percentages += (error * mask / targets.clamp_min(1e-6)).sum(
            (0, 2), dtype=torch.float64
        )
    if not torch.isfinite(absolute).all() or (counts == 0).any():
        raise RuntimeError("Invalid predictions or missing observations for a horizon")
    return {
        "mae_all": (absolute.sum() / counts.sum()).item(),
        "horizons": [
            {
                "minutes": (idx + 1) * 5,
                "mae_mph": absolute[idx].item() / counts[idx].item(),
                "rmse_mph": (squares[idx] / counts[idx]).sqrt().item(),
                "mape": (percentages[idx] / counts[idx]).item(),
            }
            for idx in (2, 5, 11)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "cst"), required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--locations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pilot-updates", type=int, default=0)
    args = parser.parse_args()
    if min(args.atoms, args.epochs, args.batch_size) <= 0 or args.pilot_updates < 0:
        parser.error("Expected positive atom count, epoch count and batch size")
    if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090" and args.pilot_updates:
        print("Pilot is running on", torch.cuda.get_device_name(), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    speeds, clock, (train, validation, test), config = training_data(
        args.data, args.locations, args.atoms, "cuda"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model = build_model(args.variant, config)
    digest = shared_hash(model)
    model.cuda()
    optimizer = make_optimizer(model, args)
    torch.manual_seed(args.seed)
    shuffle = torch.Generator(device="cuda").manual_seed(args.seed)
    graph_params = sum(p.numel() for p in model.gc.parameters())
    result = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "precision": "FP32 with TF32 enabled",
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "torchcst_commit": "1750c5ea6cbefe3d4a7d8f6ba02fd9272df3cf98",
        "initial_learning_rates": [g["lr"] for g in optimizer.param_groups],
        "shared_initial_parameters_sha256": digest,
        "graph_parameters": graph_params,
        "all_parameters": sum(p.numel() for p in model.parameters()),
        "config": config,
    }
    started = perf_counter()
    if args.pilot_updates:
        order = torch.randperm(len(train), generator=shuffle, device="cuda")
        batches = train[order].split(args.batch_size)
        if len(batches) < args.pilot_updates + 10:
            parser.error("pilot must fit in one epoch with ten warmup updates")
        timings = []
        for count, origins in enumerate(batches[: args.pilot_updates + 10]):
            inputs, targets = get_batch(speeds, clock, origins, config)
            torch.cuda.synchronize()
            if count == 10:
                torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            loss = step(model, optimizer, inputs, targets, config)
            torch.cuda.synchronize()
            if count >= 10:
                timings.append((perf_counter() - start) * 1000)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite pilot loss")
        if args.variant == "cst":
            gradient = model.gc.linear.atoms.p.grad
            if (
                gradient is None
                or not gradient.isfinite().all()
                or not gradient.abs().max() > 0
            ):
                raise RuntimeError("Invalid atom gradient")
        result["pilot"] = {
            "updates": len(timings),
            "median_step_ms": median(timings),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "last_loss_mph": loss.item(),
        }
        print(json.dumps(result["pilot"]), flush=True)
    else:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        history = []
        best = float("inf")
        for epoch in range(1, args.epochs + 1):
            total_loss = speeds.new_zeros(())
            order = torch.randperm(len(train), generator=shuffle, device="cuda")
            for indices in train[order].split(args.batch_size):
                inputs, targets = get_batch(speeds, clock, indices, config)
                total_loss += step(model, optimizer, inputs, targets, config) * len(
                    indices
                )
            score = measure(model, speeds, clock, validation, config, args.batch_size)
            row = {
                "epoch": epoch,
                "train_mae": total_loss.item() / len(train),
                "validation": score,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if score["mae_all"] < best:
                best = score["mae_all"]
                result["best_epoch"] = epoch
                staging = args.output / "best.tmp"
                torch.save(model.state_dict(), staging)
                staging.replace(args.output / "best.pt")
            schedule.step()
        result["history"] = history
        model.load_state_dict(
            torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
        )
        result["test"] = measure(model, speeds, clock, test, config, args.batch_size)
        print(
            json.dumps({"best_epoch": result["best_epoch"], "test": result["test"]}),
            flush=True,
        )
    result["elapsed_s"] = perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
