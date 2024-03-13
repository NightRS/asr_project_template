import json
import logging
import os
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable
from itertools import islice, repeat
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import torch
import tqdm

_logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(fname: Path):
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: Path):
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader: Iterable[Any]) -> Generator[Any, None, None]:
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def loop_exactly_n_times(data_loader: Iterable[Any], n: int) -> Iterable[Any]:
    return islice(inf_loop(data_loader), n)


def prepare_device(n_gpu_use: int):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        _logger.warning(
            "There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        _logger.warning(
            f"The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys: str):
        self._data = pd.DataFrame(
            {
                "total": pd.Series(0, dtype="float", index=keys),
                "counts": pd.Series(0, dtype="int", index=keys),
                "average": pd.Series(0, dtype="float", index=keys),
            }
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key: str, value: float, n: int = 1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key: str) -> float:
        return self._data.average[key]

    def result(self) -> dict[str, float]:
        return self._data.average.to_dict()

    def keys(self) -> Iterable[str]:
        return self._data.total.keys()


# https://speechbrain.readthedocs.io/en/v0.5.15/_modules/speechbrain/utils/data_utils.html#download_file
def download_file(source: str, dest: Path, replace_existing: bool = False):
    class DownloadProgressBar(tqdm.tqdm):
        def update_to(self, b: int, bsize: int, tsize: int):
            self.total = tsize
            self.update(b * bsize - self.n)

    dest_dir = dest.resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(dest) or (os.path.isfile(dest) and replace_existing):
        _logger.info(f"Downloading {source} to {dest}")
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=source.split("/")[-1],
        ) as t:
            urllib.request.urlretrieve(source, filename=dest, reporthook=t.update_to)
    else:
        _logger.info(f"{dest} exists. Skipping download")
