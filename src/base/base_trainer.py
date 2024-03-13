from abc import abstractmethod
from typing import Any, Optional

import torch
from numpy import inf

from src.base.base_metric import BaseMetric
from src.base.base_model import BaseModel
from src.logger.cloud_saver import get_cloud_saver
from src.logger.visualization import get_visualizer
from src.utils.parse_config import (
    ConfigParser,
    ResumeCloud,
    ResumeLocal,
    ResumeSettings,
)
from src.utils.util import read_json, write_json


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model: BaseModel | torch.nn.DataParallel[BaseModel],
        criterion: torch.nn.Module,
        metrics: list[BaseMetric],
        optimizer: torch.optim.Optimizer,
        config: ConfigParser,
        device: torch.device,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        self._model = model
        self._criterion = criterion
        self._metrics = metrics
        self._optimizer = optimizer
        self._config = config
        self._device = device
        self._lr_scheduler = lr_scheduler

        self._logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        cfg_trainer = config["trainer"]
        self._epochs = cfg_trainer["epochs"]
        self._save_period = cfg_trainer["save_period"]
        self._monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self._monitor == "off":
            self._mnt_mode = "off"
            self._mnt_best = 0
        else:
            self._mnt_mode, self._mnt_metric = self._monitor.split()
            assert self._mnt_mode in ["min", "max"]

            self._mnt_best = inf if self._mnt_mode == "min" else -inf
            self._early_stop = cfg_trainer.get("early_stop", inf)
            if self._early_stop <= 0:
                self._early_stop = inf

        self._start_epoch = 1
        self._steps_done = 0

        self._checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self._writer = get_visualizer(config, cfg_trainer["visualize"])

        self._cloud_saver = get_cloud_saver(cfg_trainer["cloud_saver"])
        self._cloud_dir = config.cloud_dir
        self._cloud_saver.prepare_log_folder(config.cloud_dir)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _skip_epochs(self, n_epoch: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self._logger.info("Keyboard interrupt, aborting training")
            raise e
        finally:
            self._cloud_saver.ensure_files_upload()
            self._writer.finish()

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self._start_epoch, self._epochs + 1):
            result = self._train_epoch(epoch)
            self._steps_done += result["steps_done"]

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self._logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self._mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self._mnt_mode == "min":
                        improved = log[self._mnt_metric] <= self._mnt_best
                    elif self._mnt_mode == "max":
                        improved = log[self._mnt_metric] >= self._mnt_best
                    else:
                        improved = False
                except KeyError:
                    self._logger.warning(
                        f"Warning: Metric '{self._mnt_metric}' is not found. "
                        "Model performance monitoring is disabled."
                    )
                    self._mnt_mode = "off"
                    improved = False

                if improved:
                    self._mnt_best = log[self._mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self._early_stop:
                    self._logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self._early_stop)
                    )
                    break

            if epoch % self._save_period == 0 or best:
                self._save_checkpoint(epoch, is_current_best=best)

    def _save_checkpoint(
        self,
        epoch: int,
        is_current_best: Optional[bool] = False,
    ):
        """
        Saving checkpoints
        """

        def save_impl(
            state: dict[str, Any],
            config: dict[str, Any],
            dir_name: str,
            overwrite: bool,
        ):
            self._logger.info("Saving checkpoint: {} ...".format(dir_name))

            (self._checkpoint_dir / dir_name).mkdir(parents=True, exist_ok=overwrite)

            write_json(config, self._checkpoint_dir / dir_name / "config.json")
            torch.save(state, self._checkpoint_dir / dir_name / "checkpoint.pth")

            self._logger.info("Checkpoint saved. Uploading to cloud ...")
            self._cloud_saver.upload(
                self._checkpoint_dir / dir_name / "config.json",
                self._cloud_dir / dir_name / "config.json",
                overwrite=overwrite,
            )
            self._cloud_saver.upload(
                self._checkpoint_dir / dir_name / "checkpoint.pth",
                self._cloud_dir / dir_name / "checkpoint.pth",
                overwrite=overwrite,
            )

        state = {
            "arch": type(self._model).__name__,
            "epoch": epoch,
            "steps_done": self._steps_done,
            "state_dict": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "monitor_best": self._mnt_best,
            "lr_scheduler": self._lr_scheduler.state_dict(),
            "writer": self._writer.state_dict(),
        }
        config = self._config.config

        self._cloud_saver.ensure_files_upload()
        save_impl(
            state,
            config,
            f"checkpoint_epoch_{epoch}",
            overwrite=False,
        )
        if is_current_best:
            save_impl(
                state,
                config,
                "checkpoint_best",
                overwrite=True,
            )

    def _resume_checkpoint(self, resume: ResumeSettings):
        """
        Resume from saved checkpoints
        """
        if isinstance(resume, ResumeCloud):
            (self._checkpoint_dir / "checkpoint_resume").mkdir(exist_ok=True)

            resume.cloud_saver.download(
                resume.resume_path / "config.json",
                self._checkpoint_dir / "checkpoint_resume" / "config.json",
            )
            resume.cloud_saver.download(
                resume.resume_path / "/checkpoint.pth",
                self._checkpoint_dir / "checkpoint_resume" / "checkpoint.pth",
            )
            resume_path = self._checkpoint_dir / "checkpoint_resume"
        elif isinstance(resume, ResumeLocal):
            resume_path = resume.resume_path
        else:
            raise ValueError("unknown resume type")

        self._logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint_config = read_json(resume_path / "config.json")
        checkpoint = torch.load(resume_path / "checkpoint.pth", self._device)

        self._start_epoch = checkpoint["epoch"] + 1
        self._steps_done = checkpoint["steps_done"]
        self._mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint_config["arch"] != self._config["arch"]:
            self._logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self._model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint_config["optimizer"] != self._config["optimizer"]
            or checkpoint_config["lr_scheduler"] != self._config["lr_scheduler"]
        ):
            self._logger.warning(
                "Warning: optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters are not being resumed."
            )
        else:
            self._optimizer.load_state_dict(checkpoint["optimizer"])
            self._lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self._writer.load_state_dict(checkpoint["writer"])

        if (
            (
                checkpoint_config["data"]["train"]["batch_size"]
                == self._config["data"]["train"]["batch_size"]
            )
            and (
                checkpoint_config["data"]["train"]["datasets"]
                == self._config["data"]["train"]["datasets"]
            )
            and (
                checkpoint_config["trainer"]["SortaGrad"]
                == self._config["trainer"]["SortaGrad"]
            )
            and (
                checkpoint["steps_done"]
                == checkpoint["epoch"] * self._config["trainer"]["len_epoch"]
            )
        ):
            # We'll try to restore DataLoader state for full reproducibility.
            self._logger.info(
                f"The train data given in the config file appears to be the same "
                f"as from the checkpoint. Skipping {checkpoint['epoch']} epochs."
            )
            self._skip_epochs(checkpoint["epoch"])

        self._logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self._start_epoch)
        )
