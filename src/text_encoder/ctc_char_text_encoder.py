from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from pyctcdecode.decoder import build_ctcdecoder

from src.base.base_text_encoder import BaseCTCTextEncoder
from src.text_encoder import CharTextEncoder


@dataclass
class Hypothesis:
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder, BaseCTCTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: Optional[list[str]] = None):
        super().__init__(alphabet)
        vocab: list[str] = [self.EMPTY_TOK] + list(self._alphabet)
        self._ind2char: dict[int, str] = dict(enumerate(vocab))
        self._char2ind: dict[str, int] = {v: k for k, v in self._ind2char.items()}

        labels = vocab[:]
        labels[self._char2ind[self.EMPTY_TOK]] = ""
        self._decoder = build_ctcdecoder(labels)

    def ctc_decode(self, inds: Iterable[int]) -> str:
        res = ["ඞ"]
        empty_tok_ind = self._char2ind[self.EMPTY_TOK]
        for ind in inds:
            ind = int(ind)
            if ind != empty_tok_ind and self._ind2char[ind] != res[-1]:
                res.append(self._ind2char[ind])
        return "".join(res[1:]).strip()

    def ctc_beam_search(
        self,
        probs: torch.Tensor,
        probs_length: int,
        beam_size: int = 100,
    ) -> list[Hypothesis]:
        assert len(probs.shape) == 2
        _, voc_size = probs.shape
        assert voc_size == len(self._ind2char)

        res = self._decoder.decode_beams(
            probs[:probs_length].detach().numpy(), beam_width=beam_size
        )

        return [Hypothesis(res[i][0], np.exp(res[i][4])) for i in range(len(res))]

    def ctc_beam_search_slow(
        self,
        probs: torch.Tensor,
        probs_length: int,
        beam_size: int = 100,
    ) -> list[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """

        def truncate_beam(paths: dict[tuple[str, str], float]):
            return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

        def extend_and_merge(
            next_char_probs,
            src_paths: dict[tuple[str, str], float],
        ):
            new_paths: dict[tuple[str, str], float] = defaultdict(float)
            for next_char_ind, next_char_prob in enumerate(next_char_probs):
                next_char = self._ind2char[next_char_ind]
                for (text, last_char), path_prob in src_paths.items():
                    new_prefix = text if next_char == last_char else (text + next_char)
                    new_prefix = new_prefix.replace(self.EMPTY_TOK, "")
                    new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
            return new_paths

        assert len(probs.shape) == 2
        _, voc_size = probs.shape
        assert voc_size == len(self._ind2char)

        paths: dict[tuple[str, str], float] = {("", self.EMPTY_TOK): 1.0}
        for next_char_probs in probs[:probs_length]:
            paths = extend_and_merge(next_char_probs, paths)
            paths = truncate_beam(paths)

        hypos: list[Hypothesis] = [
            Hypothesis(prefix, score) for (prefix, _), score in paths.items()
        ]

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
