"""Export verified Safetensors and the METR-LA model declaration."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import save_file
from torchcst.hub import load_safetensors

from model import NODES, build_model, forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {
        variant: json.loads((args.run / variant / "metrics.json").read_text())
        for variant in ("baseline", "cst")
    }
    original, cst = results.values()
    for key in (
        "config",
        "seed",
        "epochs",
        "batch_size",
        "shared_initial_parameters_sha256",
    ):
        if original[key] != cst[key]:
            raise ValueError(f"Incompatible baseline and CST run metadata for {key}")

    args.output.mkdir(parents=True, exist_ok=False)
    config = cst["config"]
    inputs = torch.randn(2, 2, NODES, 12, generator=torch.Generator().manual_seed(17))
    with torch.inference_mode():
        for variant in results:
            factory = partial(build_model, variant, config)
            baseline = factory().eval()
            state = torch.load(
                args.run / variant / "best.pt", map_location="cpu", weights_only=True
            )
            baseline.load_state_dict(state)
            weights = {
                name: tensor.detach().clone().contiguous()
                for name, tensor in state.items()
                if isinstance(tensor, torch.Tensor)
            }
            path = args.output / f"{variant}.safetensors"
            save_file(weights, path, metadata={"variant": variant, "format": "pt"})
            model = load_safetensors(path, model_factory=factory)
            torch.testing.assert_close(
                forecast(model, inputs, config),
                forecast(baseline, inputs, config),
                rtol=0,
                atol=0,
            )
            print(variant, "Safetensors forecast matches the source checkpoint exactly")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
