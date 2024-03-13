import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate and pad fields in dataset items
    """
    return {
        "spectrogram": torch.nn.utils.rnn.pad_sequence(
            [item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items],
            batch_first=True,
        )
        .transpose(1, 2)
        .contiguous(),
        "spectrogram_length": torch.tensor(
            [item["spectrogram"].size(2) for item in dataset_items], dtype=torch.int32
        ),
        "text_encoded": torch.nn.utils.rnn.pad_sequence(
            [item["text_encoded"].squeeze(0) for item in dataset_items],
            batch_first=True,
        ),
        "text_encoded_length": torch.tensor(
            [item["text_encoded"].size(1) for item in dataset_items], dtype=torch.int32
        ),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
        "audio": [item["audio"] for item in dataset_items],
    }
