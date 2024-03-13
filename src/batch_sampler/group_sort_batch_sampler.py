from collections.abc import Iterator
from typing import Any

from torch.utils.data import Dataset, Sampler


class GroupLengthBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        data_source: Dataset[dict[str, Any]],
        batch_size: int,
        batches_per_group: int = 20,
    ):
        super().__init__()
        raise NotImplementedError()

    def __iter__(self) -> Iterator[list[int]]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
