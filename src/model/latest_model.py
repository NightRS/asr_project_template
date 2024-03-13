from typing import Any

import torch
from torch import nn

from src.base.base_model import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats: int, n_class: int, rnn_hidden: int, **kwargs: Any):
        super().__init__(n_feats, n_class, **kwargs)

        def calc_h_out(h_in: int):
            out_h1 = ((h_in + 2 * 20 - (41 - 1) - 1) // 2) + 1
            out_h2 = ((out_h1 + 2 * 10 - (21 - 1) - 1) // 2) + 1
            out_h3 = ((out_h2 + 2 * 10 - (21 - 1) - 1) // 2) + 1
            return out_h3

        self._rnn_in = 96 * calc_h_out(n_feats)
        self._rnn_hidden = rnn_hidden

        self._conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(96),
            nn.Hardtanh(0, 20, inplace=True),
        )
        self._rnn = nn.GRU(
            self._rnn_in,
            rnn_hidden,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self._fc = nn.Linear(in_features=rnn_hidden, out_features=n_class)

    def forward(self, *, spectrogram: torch.Tensor, **batch: Any) -> dict[str, Any]:
        # spectrogram: batch_size, n_mels, max_time
        # out: batch_size, transform(max_time), alphabet_size

        h = spectrogram.unsqueeze(1)
        h = self._conv(h)
        h = h.view(spectrogram.size()[0], self._rnn_in, -1).transpose(1, 2)
        h, _ = self._rnn(h)
        # batch_size, transform(max_time), rnn_hidden
        h = h[:, :, : self._rnn_hidden] + h[:, :, self._rnn_hidden :]
        h = self._fc(h)

        return {"logits": h}

    def transform_input_lengths(self, input_lengths: int) -> int:
        w_in = input_lengths
        out_w1 = ((w_in + 2 * 5 - (11 - 1) - 1) // 2) + 1
        out_w2 = ((out_w1 + 2 * 5 - (11 - 1) - 1) // 1) + 1
        out_w3 = ((out_w2 + 2 * 5 - (11 - 1) - 1) // 1) + 1
        return out_w3
