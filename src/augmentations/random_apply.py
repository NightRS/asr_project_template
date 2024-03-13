from collections.abc import Callable

import numpy as np
from torch import Tensor


class RandomApply:
    def __init__(self, augmentation: Callable[[Tensor], Tensor], p: float):
        assert 0 <= p <= 1
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        if np.random.rand() < self.p:
            return self.augmentation(data)
        else:
            return data
