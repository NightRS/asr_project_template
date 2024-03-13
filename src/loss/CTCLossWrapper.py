from typing import Any

import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(torch.nn.Module):
    def __init__(self, blank: int, zero_infinity: bool):
        super().__init__()
        self._loss_impl = CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: Tensor,
        log_probs_length: Tensor,
        text_encoded: Tensor,
        text_encoded_length: Tensor,
        **batch: Any,
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        return self._loss_impl.forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
