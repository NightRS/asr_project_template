import logging
import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from wandb.sdk.lib.runid import generate_id

import wandb
from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class WanDBWriter:
    def __init__(self, config: ConfigParser):
        wandb.login()

        if config["trainer"].get("wandb_project") is None:
            raise ValueError("please specify project name for wandb")

        self._run_name = config.run_name
        self._run_id = os.environ.get("RUN_ID_FOR_WANDB", generate_id())

        wandb.init(
            project=config["trainer"].get("wandb_project"),
            name=config.run_name,
            config=config.config,
            id=self._run_id,
            resume="must" if os.environ.get("RUN_ID_FOR_WANDB") else "never",
        )

        self._wandb = wandb

        self._step: int = -1
        self._mode: str = ""
        self._timer: datetime = datetime.now()
        self._total_time: float = 0.0
        self._log: dict[str, Any] = dict()
        self._mode_time_usage: dict[str, float] = {"": 0.0}

    def state_dict(self) -> dict[str, Any]:
        return {
            "run_name": self._run_name,
            "run_id": self._run_id,
            "step": self._step,
            "mode": self._mode,
            "timer": self._timer,
            "total_time": self._total_time,
            "log": self._log,
            "mode_time_usage": self._mode_time_usage,
        }

    def load_state_dict(self, state_dict: dict[str, Any], reset_timer: bool = True):
        if self._run_name != state_dict["run_name"]:
            logger.warning(
                f"wandb run name ({self._run_name}) "
                f"!= checkpoint wandb run name ({state_dict['run_name']})"
            )
        if self._run_id != state_dict["run_id"]:
            logger.warning(
                f"wandb run_id ({self._run_id}) "
                f"!= checkpoint wandb run_id ({state_dict['run_id']})"
            )

        self._step = state_dict["step"]
        self._mode = state_dict["mode"]
        if reset_timer:
            self._timer = datetime.now()
        else:
            self._timer = state_dict["timer"]
        self._total_time = state_dict["total_time"]
        self._log = state_dict["log"]
        self._mode_time_usage = state_dict["mode_time_usage"]

    def set_mode_names(self, names: Iterable[str]):
        for mode in names:
            if mode not in self._mode_time_usage:
                self._mode_time_usage[mode] = 0.0

    def set_step(self, step: int, mode: str = "train"):
        prev_mode, self._mode = self._mode, mode
        step_diff, self._step = step - self._step, step

        new_time = datetime.now()
        time_diff, self._timer = (new_time - self._timer).total_seconds(), new_time
        self._total_time += time_diff

        self._mode_time_usage[prev_mode] += time_diff

        log_train_steps = step_diff > 0
        log_time_usage = step_diff > 0 or self._mode == ""

        if log_train_steps:
            self._log.update(
                {
                    "steps_logged": step_diff,
                    "train_steps_per_sec": step_diff / time_diff,
                    "seconds_per_train_step": time_diff / step_diff,
                }
            )

        if log_time_usage:
            self._log.update(
                {
                    "total_time_sec": self._total_time,
                    "total_time_hr": self._total_time / 3600,
                    **{
                        f"{m}_time_usage_sec": time
                        for m, time in self._mode_time_usage.items()
                    },
                    **{
                        f"{m}_time_usage_relative": time / self._total_time
                        for m, time in self._mode_time_usage.items()
                    },
                }
            )

    def add_scalars(self, scalars: dict[str, float]):
        self._log.update(
            {
                **{
                    f"{self._mode}_metrics/{scalar_name}": scalar
                    for scalar_name, scalar in scalars.items()
                }
            }
        )

    def log_manual(self, data: Any, step: int):
        self._wandb.log(data, step=step)

    def write(self):
        self._wandb.log(self._log, step=self._step)
        self._log = dict()

    def _scalar_name(self, scalar_name: str) -> str:
        return f"{scalar_name}_{self._mode}"

    def add_image(
        self,
        scalar_name: str,
        image: Any,
        caption: Optional[str] = None,
    ):
        self._wandb.log(
            {self._scalar_name(scalar_name): self._wandb.Image(image, caption=caption)},
            step=self._step,
        )

    def add_audio(
        self,
        scalar_name: str,
        audio: torch.Tensor,
        sample_rate: Optional[float] = None,
        caption: Optional[str] = None,
    ):
        audio = audio.detach().cpu().numpy().T
        self._wandb.log(
            {
                self._scalar_name(scalar_name): self._wandb.Audio(
                    audio,
                    sample_rate=sample_rate,
                    caption=caption,
                )
            },
            step=self._step,
        )

    def add_text(
        self,
        scalar_name: str,
        text: str,
    ):
        self._wandb.log(
            {self._scalar_name(scalar_name): self._wandb.Html(text)}, step=self._step
        )

    def add_histogram(
        self,
        scalar_name: str,
        hist: torch.Tensor,
        bins: int | str = "auto",
    ):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        wandb_hist = self._wandb.Histogram(np_histogram=np_hist)

        self._wandb.log({self._scalar_name(scalar_name): wandb_hist}, step=self._step)

    def add_table(
        self,
        table_name: str,
        table: pd.DataFrame,
    ):
        self._wandb.log(
            {
                self._scalar_name(table_name)
                + f"/{self._step}": wandb.Table(dataframe=table)
            },
            step=self._step,
        )

    def finish(self):
        self._wandb.finish()
