from torch import Tensor
import torchaudio

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=max_length)

    def __call__(self, data: Tensor):
        return self._aug(data)
