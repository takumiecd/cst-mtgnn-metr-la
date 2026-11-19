"""Plot validation and training MAE for an MTGNN comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure, (train, valid) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    figure.suptitle("MTGNN traffic forecasting on METR-LA")
    for variant, label, color, linestyle in (
        ("baseline", "Original MTGNN", "#175d95", "-"),
        ("cst", "CST graph", "#af480c", "--"),
    ):
        record = json.loads((args.run / variant / "metrics.json").read_text())
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        style = dict(label=label, color=color, linestyle=linestyle)
        train.plot(epochs, [row["train_mae"] for row in history], **style)
        valid.plot(epochs, [row["validation"]["mae_all"] for row in history], **style)
        best = record["best_epoch"]
        valid.scatter(
            best, history[best - 1]["validation"]["mae_all"], color=color, s=25
        )
    train.set_title("Training")
    valid.set_title("Validation")
    for axis in (train, valid):
        axis.set(xlabel="Epoch", ylabel="Masked MAE (mph)")
        axis.grid(axis="y", color="#d8e0e7", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    train.legend(frameon=False)
    figure.savefig(args.output, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
