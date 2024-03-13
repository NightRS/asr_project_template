import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

import src.datasets
from src import batch_sampler as batch_sampler_module
from src.base.base_dataset import BaseDataset
from src.base.base_text_encoder import BaseTextEncoder
from src.collate_fn.collate import collate_fn
from src.datasets import CustomAudioDataset
from src.utils.parse_config import ConfigParser


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class DataloaderParams:
    batch_size: int
    shuffle: bool
    batch_sampler: Optional[Sampler[list[int]]]
    num_workers: int
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]]
    drop_last: bool
    worker_init_fn: Callable[[int], None]
    generator: torch.Generator


def get_dataloader_from_params(
    dataset: BaseDataset, params: DataloaderParams
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        batch_sampler=params.batch_sampler,
        num_workers=params.num_workers,
        collate_fn=params.collate_fn,
        drop_last=params.drop_last,
        worker_init_fn=params.worker_init_fn,
        generator=params.generator,
    )


def get_datasets(
    configs: ConfigParser,
    text_encoder: BaseTextEncoder,
) -> dict[str, tuple[BaseDataset, DataloaderParams]]:
    assert "train" in configs["data"], "You must provide exactly one train dataset"
    assert len(
        [val_name for val_name in configs["data"] if val_name != "train"]
    ), "You must provide at least one evaluation dataset"

    all_datasets: dict[str, tuple[BaseDataset, DataloaderParams]] = {}
    for split, params in configs["data"].items():
        datasets: list[BaseDataset] = []

        for ds in params["datasets"]:
            datasets.append(
                configs.init_obj(
                    ds,
                    src.datasets,
                    text_encoder=text_encoder,
                    config_parser=configs,
                )
            )

        dataset = CustomAudioDataset(
            sum([d._index for d in datasets], []),
            text_encoder=text_encoder,
            config_parser=configs,
        )

        if "batch_size" in params:
            batch_size = params["batch_size"]
            shuffle = True
            batch_sampler = None
        elif "batch_sampler" in params:
            batch_size = 1
            shuffle = False
            batch_sampler = configs.init_obj(
                params["batch_sampler"], batch_sampler_module, data_source=dataset
            )
        else:
            raise ValueError(
                "You must provide batch_size or batch_sampler for each split"
            )

        assert batch_size <= len(
            dataset
        ), f"Batch size ({batch_size}) shouldn't be larger than dataset length ({len(dataset)})"

        num_workers = params.get("num_workers", 1)

        all_datasets[split] = (
            dataset,
            DataloaderParams(
                batch_size=batch_size,
                shuffle=shuffle,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=split == "train",
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(0),
            ),
        )
    return all_datasets
