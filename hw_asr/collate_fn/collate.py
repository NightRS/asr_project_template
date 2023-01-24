import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    feature_length_dim = dataset_items[0]["spectrogram"].size(1)
    time_legths = [item["spectrogram"].size(2) for item in dataset_items]
    time_dim = max(time_legths)
    batch_spectrogram = torch.zeros(len(dataset_items), feature_length_dim, time_dim)
    for i, item in enumerate(dataset_items):
        batch_spectrogram[i, :, :time_legths[i]] = item["spectrogram"]

    text = [item["text"] for item in dataset_items]
    lengths = [item["text_encoded"].size(1) for item in dataset_items]
    text_encoded_length = torch.tensor(lengths)
    text_encoded = torch.zeros(len(dataset_items), max(lengths))
    for i, item in enumerate(dataset_items):
        text_encoded[i, :lengths[i]] = item["text_encoded"]

    return {
        "spectrogram": batch_spectrogram,
        "spectrogram_length": torch.tensor(time_legths),
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_length,
        "text": text,
        "audio_path": [item["audio_path"] for item in dataset_items],
        "audio": [item["audio"] for item in dataset_items],
    }
