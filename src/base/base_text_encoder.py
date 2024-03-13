import re
from collections.abc import Iterable

from torch import Tensor


class BaseTextEncoder:
    def encode(self, text: str) -> Tensor:
        raise NotImplementedError()

    def decode(self, vector: Iterable[int]) -> str:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, item: int) -> str:
        raise NotImplementedError()

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class BaseCTCTextEncoder(BaseTextEncoder):
    def ctc_decode(self, inds: Iterable[int]) -> str:
        raise NotImplementedError()
