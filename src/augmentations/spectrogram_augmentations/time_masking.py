from typing import Any

import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class TimeMasking(AugmentationBase):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._aug = torchaudio.transforms.TimeMasking(**kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)


class RandomTimeMasking(AugmentationBase):
    def __init__(self, *, p: float, **kwargs: Any):
        super().__init__()
        self._aug = RandomApply(
            TimeMasking(**kwargs),
            p=p,
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
