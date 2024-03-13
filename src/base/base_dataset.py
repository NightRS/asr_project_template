import logging
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.base.base_text_encoder import BaseTextEncoder
from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        index: list[dict[str, Any]],
        text_encoder: BaseTextEncoder,
        config_parser: ConfigParser,
        wave_augs: Optional[Callable[[Tensor], Tensor]] = None,
        spec_augs: Optional[Callable[[Tensor], Tensor]] = None,
        limit: Optional[int] = None,
        max_audio_length: Optional[float] = None,
        max_text_length: Optional[float] = None,
    ):
        self._text_encoder = text_encoder
        self._config_parser = config_parser
        self._wave_augs = wave_augs
        self._spec_augs = spec_augs
        self._log_spec: bool = config_parser["preprocessing"]["log_spec"]

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length, limit
        )
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: list[dict[str, Any]] = index

        # for skipping DataLoader steps
        self._dummy_flag = False

    @staticmethod
    def _sort_index(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(index, key=lambda x: x["audio_len"])

    def __getitem__(self, ind: int) -> dict[str, Any]:
        if self._dummy_flag:
            return {
                "audio": torch.zeros(1),
                "spectrogram": torch.zeros(1, 1, 1),
                "duration": 0.0,
                "text": "",
                "text_encoded": torch.zeros(1, 1),
                "audio_path": "",
            }

        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self._load_audio(audio_path)
        audio_wave, audio_spec = self._process_wave(audio_wave)

        return {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self._config_parser["preprocessing"]["sr"],
            "text": data_dict["text"],
            "text_encoded": self._text_encoder.encode(data_dict["text"]),
            "audio_path": audio_path,
        }

    def __len__(self) -> int:
        return len(self._index)

    def set_augs(
        self,
        wave_augs: Optional[Callable[[Tensor], Tensor]] = None,
        spec_augs: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        self._wave_augs = wave_augs
        self._spec_augs = spec_augs

    def set_dummy_flag(self, dummy_flag: bool):
        self._dummy_flag = dummy_flag

    def _load_audio(self, path: str) -> Tensor:
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self._config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @torch.no_grad()
    def _process_wave(self, audio_tensor_wave: Tensor) -> tuple[Tensor, Tensor]:
        if self._wave_augs is not None:
            audio_tensor_wave = self._wave_augs(audio_tensor_wave)
        wave2spec = self._config_parser.init_obj(
            self._config_parser["preprocessing"]["spectrogram"],
            torchaudio.transforms,
        )
        audio_tensor_spec = wave2spec(audio_tensor_wave)
        if self._spec_augs is not None:
            audio_tensor_spec = self._spec_augs(audio_tensor_spec)
        if self._log_spec:
            audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
        return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
        max_audio_length: Optional[float],
        max_text_length: Optional[float],
        limit: Optional[int],
    ) -> list[dict[str, Any]]:
        initial_size = len(index)

        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = np.zeros(initial_size, dtype=bool)

        if max_text_length is not None:
            exceeds_text_length = (
                np.array(
                    [len(BaseTextEncoder.normalize_text(el["text"])) for el in index]
                )
                >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = np.zeros(initial_size, dtype=bool)

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            np.random.default_rng(42).shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]]):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )
