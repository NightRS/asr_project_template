from typing import Any

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import calc_wer
from src.text_encoder import CTCCharTextEncoder


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, *, text_encoder: BaseTextEncoder, **kwargs: Any):
        super().__init__(**kwargs)
        self._text_encoder = text_encoder

    def __call__(
        self,
        *,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text: list[str],
        **kwargs: Any,
    ) -> float:
        wers: list[float] = []
        predictions = torch.argmax(log_probs.detach().cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if isinstance(self._text_encoder, CTCCharTextEncoder):
                pred_text = self._text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self._text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
