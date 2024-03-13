from typing import Any

import torch_audiomentations
from torch import Tensor

from src.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._aug = torch_audiomentations.Gain(**kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
