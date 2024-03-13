from typing import Any

import torchaudio
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class FrequencyMasking(AugmentationBase):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(**kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data)


class RandomFrequencyMasking(AugmentationBase):
    def __init__(self, *, p: float, **kwargs: Any):
        super().__init__()
        self._aug = RandomApply(
            FrequencyMasking(**kwargs),
            p=p,
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
