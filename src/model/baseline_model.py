from typing import Any

import torch
from torch import nn
from torch.nn import Sequential

from src.base.base_model import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats: int, n_class: int, fc_hidden: int = 512, **kwargs: Any):
        super().__init__(n_feats, n_class, **kwargs)
        self._net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, *, spectrogram: torch.Tensor, **batch: Any) -> dict[str, Any]:
        return {"logits": self._net(spectrogram.transpose(1, 2))}

    def transform_input_lengths(self, input_lengths: int) -> int:
        return input_lengths  # we don't reduce time dimension here

    def classify_parameter_name(self, prefix: str, name: str) -> str:
        if name.startswith("net.0"):
            return f"{prefix}_0/{name}"
        elif name.startswith("net.2"):
            return f"{prefix}_2/{name}"
        elif name.startswith("net.4"):
            return f"{prefix}_4/{name}"
        return f"{prefix}/{name}"
