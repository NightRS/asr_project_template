import json
from collections.abc import Iterable
from pathlib import Path
from string import ascii_lowercase
from typing import Optional

from torch import Tensor

from src.base.base_text_encoder import BaseTextEncoder


class CharTextEncoder(BaseTextEncoder):
    def __init__(self, alphabet: Optional[list[str]] = None):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")
        self._alphabet: list[str] = alphabet
        self._ind2char: dict[int, str] = {k: v for k, v in enumerate(sorted(alphabet))}
        self._char2ind: dict[str, int] = {v: k for k, v in self._ind2char.items()}

    def __len__(self) -> int:
        return len(self._ind2char)

    def __getitem__(self, item: int) -> str:
        return self._ind2char[item]

    def encode(self, text: str) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor([self._char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self._char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, vector: Iterable[int]) -> str:
        return "".join([self._ind2char[int(ind)] for ind in vector]).strip()

    def dump(self, file: str):
        with Path(file).open("w") as f:
            json.dump(self._ind2char, f)

    @classmethod
    def from_file(cls, file: str):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a._ind2char = ind2char
        a._char2ind = {v: k for k, v in ind2char}
        return a
