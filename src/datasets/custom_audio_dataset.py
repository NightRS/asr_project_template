import logging
from pathlib import Path
from typing import Any

import torchaudio

from src.base.base_dataset import BaseDataset
from src.utils.util import read_json

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data: list[dict[str, Any]], *args: Any, **kwargs: Any):
        for entry in data:
            assert "path" in entry
            assert Path(entry["path"]).exists(), f"Path {entry['path']} doesn't exist"
            entry["path"] = str(Path(entry["path"]).absolute().resolve())
            entry["text"] = entry.get("text", "")
            t_info = torchaudio.info(entry["path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(data, *args, **kwargs)
