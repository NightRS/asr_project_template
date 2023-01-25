from torch import Tensor
import torchaudio

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = torchaudio.transforms.TimeStretch(fixed_rate=rate)

    def __call__(self, data: Tensor):
        return self._aug(data)
