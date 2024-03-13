from abc import abstractmethod
from typing import Any

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self, n_feats: int, n_class: int, **kwargs: Any):
        super().__init__()

    @abstractmethod
    def forward(self, **batch: Any) -> dict[str, Any]:
        """
        Forward pass logic.
        Can return a torch.Tensor (it will be interpreted as logits) or a dict.

        :return: Model output
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def transform_input_lengths(self, input_lengths: int) -> int:
        """
        Input length transformation function.
        For example: if your NN transforms spectrogram of time-length `N` into an
            output with time-length `N / 2`, then this function should return `input_lengths // 2`
        """
        raise NotImplementedError()

    def classify_parameter_name(self, prefix: str, name: str) -> str:
        """
        Obtain a folder name for further logging of model parameters.
        """
        raise NotImplementedError()
