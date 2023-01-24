from typing import List, NamedTuple

from collections import defaultdict

#from pyctcdecode import build_ctcdecoder

from ctcdecode import CTCBeamDecoder

import torch
import numpy as np

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        # for pyctcdecode

        # labels = list(self.ind2char.values())
        # labels[0] = ''
        # self.decoder = build_ctcdecoder(labels)

    def ctc_decode(self, inds: List[int]) -> str:
        res = ['ඞ']
        for ind in inds:
            # self.char2ind[EMPTY_TOK] == 0
            if ind != 0 and self.ind2char[ind] != res[-1]:
                res.append(self.ind2char[ind])
        return ''.join(res[1:])

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        assert len(probs.shape) == 2
        _, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        decoder = CTCBeamDecoder(list(self.ind2char.values()), beam_width=beam_size)
        beam_results, beam_scores, _, out_lens = decoder.decode(torch.unsqueeze(probs[:probs_length], 0))
        prob = 1/beam_scores.exp()

        return [Hypothesis(self.decode(beam_results[0, i][:out_lens[0, i]]), prob[0, i].item()) for i in range(beam_size)]

    def ctc_beam_search_pyctcdecode(self, probs: torch.tensor, probs_length,
                                    beam_size: int = 100) -> List[Hypothesis]:
        assert len(probs.shape) == 2
        _, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        res = self.decoder.decode_beams(probs[:probs_length].detach().numpy(), beam_width=beam_size)

        return [Hypothesis(res[i][0], np.exp(res[i][3])) for i in range(len(res))]

    def ctc_beam_search_slow(self, probs: torch.tensor, probs_length,
                             beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        def truncate_beam(paths):
            return dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

        def extend_and_merge(next_char_probs, src_paths):
            new_paths = defaultdict(float)
            for next_char_ind, next_char_prob in enumerate(next_char_probs):
                next_char = self.ind2char[next_char_ind]
                for (text, last_char), path_prob in src_paths.items():
                    new_prefix = text if next_char == last_char else (text + next_char)
                    new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                    new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
            return new_paths

        assert len(probs.shape) == 2
        _, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        paths = {('', self.EMPTY_TOK): 1.0}
        for next_char_probs in probs[:probs_length]:
            paths = extend_and_merge(next_char_probs, paths)
            paths = truncate_beam(paths)

        hypos: List[Hypothesis] = [Hypothesis(prefix, score) for (prefix, _), score in paths.items()]

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
