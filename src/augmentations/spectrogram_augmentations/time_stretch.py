from typing import Any

import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class TimeStretch(AugmentationBase):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._aug = torchaudio.transforms.TimeStretch(**kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)


class RandomTimeStretch(AugmentationBase):
    def __init__(self, *, p: float, **kwargs: Any):
        super().__init__()
        self._aug = RandomApply(
            TimeStretch(**kwargs),
            p=p,
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
