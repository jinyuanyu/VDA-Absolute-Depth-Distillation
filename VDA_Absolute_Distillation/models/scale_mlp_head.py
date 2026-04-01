from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class ScaleMLPHead(nn.Module):
    """Predict per-image scale and shift from a global feature vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (512, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        dims: List[int] = [input_dim, *hidden_dims, 2]
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    @staticmethod
    def split_scale_shift(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = prediction[..., 0]
        shift = prediction[..., 1]
        return scale, shift
