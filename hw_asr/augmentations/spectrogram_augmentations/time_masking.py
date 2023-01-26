from torch import Tensor
import torchaudio

from hw_asr.augmentations.base import AugmentationBase

from hw_asr.augmentations.random_apply import RandomApply


class TimeMasking(AugmentationBase):
    def __init__(self, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param=max_length)

    def __call__(self, data: Tensor):
        return self._aug(data)


class RandomTimeMasking:
    def __init__(self, p, max_length):
        self._aug = RandomApply(
            TimeMasking(max_length=max_length),
            p=p,
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
