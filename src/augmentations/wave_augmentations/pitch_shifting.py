from functools import partial
from typing import Any, Callable

import librosa
import torch
from numpy import ndarray
from torch import Tensor

from src.augmentations.base import AugmentationBase
from src.augmentations.random_apply import RandomApply


class PitchShifting(AugmentationBase):
    def __init__(self, sr: int, n_steps: float):
        super().__init__()
        self._aug: Callable[[ndarray], ndarray] = partial(
            librosa.effects.pitch_shift,
            sr=sr,
            n_steps=n_steps,
        )

    def __call__(self, data: Tensor):
        x = self._aug(data.numpy().squeeze())
        return torch.from_numpy(x)


class RandomPitchShifting:
    def __init__(self, p: float, **kwargs: Any):
        self._aug = RandomApply(
            PitchShifting(**kwargs),
            p=p,
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
