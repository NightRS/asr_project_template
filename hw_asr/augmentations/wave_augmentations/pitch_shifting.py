import librosa
import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase

from hw_asr.augmentations.random_apply import RandomApply


class PitchShifting(AugmentationBase):
    def __init__(self, sr, n_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = librosa.effects.pitch_shift
        self.sr = sr
        self.n_steps = n_steps

    def __call__(self, data: Tensor):
        x = self._aug(data.numpy().squeeze(), self.sr, self.n_steps)
        return torch.from_numpy(x)


class RandomPitchShifting:
    def __init__(self, p, sr, n_steps):
        self._aug = RandomApply(
            PitchShifting(sr=sr, n_steps=n_steps),
            p=p,
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
