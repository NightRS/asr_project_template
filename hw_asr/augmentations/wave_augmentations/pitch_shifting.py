import librosa
import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShifting(AugmentationBase):
    def __init__(self, sr, n_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = librosa.effects.pitch_shift
        self.sr = sr
        self.n_steps = n_steps

    def __call__(self, data: Tensor):
        x = self._aug(data.numpy().squeeze(), self.sr, self.n_steps)
        return torch.from_numpy(x)
