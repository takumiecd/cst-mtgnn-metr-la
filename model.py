"""MTGNN with an optional continuous parameterization of graph scores."""

from __future__ import annotations

import torch
from torch import nn

from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, chart_presets, kernels
from upstream.layer import graph_constructor, retain_topk
from upstream.net import gtnet

NODES = 207
HORIZON = 12
TOP_K = 20
ALPHA = 3.0


class CSTGraph(nn.Module):
    def __init__(self, coordinates, atoms: int):
        super().__init__()
        chart = chart_presets.points(coordinates)
        kernel = kernels.product(
            kernels.l2(TriweightSpec()),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.1,
                bounds=BandwidthBounds(
                    minimum=0.5, birth=3.5, maximum=6.0, upper_floor=0.5
                ),
            ),
        )
        self.linear = CSTLinear(
            charts=(chart, chart), atoms=atoms, kernel=kernel, backend="factored"
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        input_factors, output_factors = self.linear.operator.factors()
        scores = output_factors[idx] @ input_factors[idx].transpose(0, 1)
        adjacency = (ALPHA * (scores - scores.transpose(0, 1))).tanh().relu()
        return retain_topk(adjacency, TOP_K, self.training)


def build_model(variant: str, config: dict) -> nn.Module:
    model = gtnet(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=NODES,
        device="cpu",
        subgraph_size=TOP_K,
        tanhalpha=ALPHA,
    )
    if variant == "cst":
        model.gc = CSTGraph(config["coordinates_km"], config["atoms"])
    elif variant == "baseline_coords":
        coordinates = torch.tensor(config["coordinates_km"], dtype=torch.float32)
        # Match the scale of the original unit-variance node embeddings.
        coordinates = coordinates / coordinates.square().mean().sqrt().clamp_min(1e-6)
        model.gc = graph_constructor(
            NODES, TOP_K, model.gc.dim, "cpu",
            alpha=ALPHA, static_feat=coordinates, append_static=True,
        )
    return model


def forecast(model: nn.Module, inputs: torch.Tensor, config: dict) -> torch.Tensor:
    return model(inputs).squeeze(-1) * config["std_mph"] + config["mean_mph"]
