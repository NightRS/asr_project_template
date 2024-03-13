import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.augmentations
from src.base.base_dataset import BaseDataset
from src.base.base_metric import BaseMetric
from src.base.base_model import BaseModel
from src.base.base_text_encoder import BaseCTCTextEncoder, BaseTextEncoder
from src.base.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrograms_to_buf
from src.metric.utils import calc_cer, calc_wer
from src.utils.object_loading import DataloaderParams, get_dataloader_from_params
from src.utils.parse_config import ConfigParser
from src.utils.util import MetricTracker, loop_exactly_n_times


class Trainer(BaseTrainer):
    """
    Trainer class
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
        dataset_and_params: dict[str, tuple[BaseDataset, DataloaderParams]],
        text_encoder: BaseCTCTextEncoder,
    ):
        self._dataset_and_params = dataset_and_params
        self._text_encoder = text_encoder

        train_dataset, train_params = dataset_and_params["train"]
        self._train_dataloader = get_dataloader_from_params(train_dataset, train_params)
        self._sortagrad_train_dataloader = get_dataloader_from_params(
            train_dataset,
            dataclasses.replace(
                train_params,
                shuffle=False,
            ),
        )

        self._evaluation_dataloaders = {
            split: get_dataloader_from_params(dataset, params)
            for split, (dataset, params) in dataset_and_params.items()
            if split != "train"
        }

        self._len_epoch = config["trainer"]["len_epoch"]

        self._wave_augs, self._spec_augs = src.augmentations.from_configs(config)

        self._log_step = config["trainer"]["log_step"]
        self._media_log_step = config["trainer"]["media_log_step"]
        self._grad_norm_clip_threshold = config["trainer"]["grad_norm_clip_threshold"]

        super().__init__(
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            lr_scheduler,
        )

        self._writer.set_mode_names(dataset_and_params.keys())

        self._train_metrics = MetricTracker(
            "loss",
            "grad_norm",
            *[m.name for m in self._metrics],
        )
        self._evaluation_metrics = MetricTracker(
            "loss",
            *[m.name for m in self._metrics],
        )

    @staticmethod
    def move_batch_to_device(
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _skip_epochs(self, n_epoch: int):
        self._dataset_and_params["train"][0].set_dummy_flag(True)

        for epoch in range(1, n_epoch + 1):
            train_dataloader = self._train_dataloader
            if epoch == 1 and self._config["trainer"]["SortaGrad"]:
                train_dataloader = self._sortagrad_train_dataloader
            train_dataloader = loop_exactly_n_times(train_dataloader, n=self._len_epoch)
            for _ in tqdm(train_dataloader, desc="train_skip", total=self._len_epoch):
                pass

        self._dataset_and_params["train"][0].set_dummy_flag(False)

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        def writer_start_train_epoch():
            self._train_metrics.reset()
            self._writer.set_step(self._steps_done - 1, "train")
            self._writer.log_manual(
                {"train_metrics/epoch": epoch}, step=self._steps_done
            )

        def writer_log_scalars() -> dict[str, float]:
            self._writer.add_scalars(
                {
                    "epoch": epoch,
                    "learning_rate": self._lr_scheduler.get_last_lr()[0],
                }
            )
            res = self._train_metrics.result()
            self._writer.add_scalars(res)
            self._writer.write()
            self._train_metrics.reset()
            return res

        def writer_log_media():
            self._log_model_weights()
            self._log_model_gradients()

        def writer_log_if_needed(
            batch_idx: int,
            batch: dict[str, Any],
            last_res: dict[str, Any],
        ) -> dict[str, Any]:
            if (
                (batch_idx + 1) % self._log_step == 0
                or (batch_idx + 1) % self._media_log_step == 0
                or (batch_idx + 1) == self._len_epoch
            ):
                self._writer.set_step(self._steps_done + batch_idx, "train")

            if (
                #
                (batch_idx + 1) % self._log_step == 0
                or (batch_idx + 1) == self._len_epoch
            ):
                last_res = writer_log_scalars()

            if (batch_idx + 1) % self._media_log_step == 0:
                writer_log_media()

            if batch_idx == 0:
                self._log_train_spectrograms(**batch)

            return last_res

        def writer_end_train_epoch():
            self._writer.set_step(self._steps_done + self._len_epoch - 1, "")
            self._writer.write()

        train_dataloader = self._train_dataloader
        if epoch == 1 and self._config["trainer"]["SortaGrad"]:
            train_dataloader = self._sortagrad_train_dataloader

        if epoch >= self._config["trainer"]["augs_start_epoch"]:
            self._dataset_and_params["train"][0].set_augs(
                wave_augs=self._wave_augs,
                spec_augs=self._spec_augs,
            )

        train_dataloader = loop_exactly_n_times(train_dataloader, n=self._len_epoch)

        self._model.train()
        writer_start_train_epoch()

        last_train_metrics = {}
        with tqdm(train_dataloader, desc="train", total=self._len_epoch) as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = self._process_batch(
                    batch,
                    is_train=True,
                    metrics=self._train_metrics,
                )
                pbar.set_postfix_str(f"Loss: {batch['loss'].item():.4f}")
                last_train_metrics = writer_log_if_needed(
                    batch_idx,
                    batch,
                    last_train_metrics,
                )

        log = last_train_metrics
        log.update({"steps_done": self._len_epoch})
        for part, dataloader in self._evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        writer_end_train_epoch()
        return log

    def _process_batch(
        self,
        batch: dict[str, Any],
        is_train: bool,
        metrics: MetricTracker,
    ) -> dict[str, Any]:
        batch = self.move_batch_to_device(batch, self._device)
        if is_train:
            self._optimizer.zero_grad()

        outputs = self._model(**batch)
        batch["logits"] = outputs

        # (batch_size, transform(max_time), alphabet_size)
        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self._model.transform_input_lengths(
            batch["spectrogram_length"]
        )
        batch["loss"] = self._criterion(**batch)

        if is_train:
            batch["loss"].backward()
            grad_norm = clip_grad_norm_(
                self._model.parameters(),
                self._grad_norm_clip_threshold,
            )
            metrics.update("grad_norm", grad_norm.item())
            self._optimizer.step()
            self._lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self._metrics:
            metrics.update(met.name, met(**batch))
        return batch

    @torch.no_grad()
    def _evaluation_epoch(
        self,
        part: str,
        dataloader: DataLoader[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self._model.eval()
        self._evaluation_metrics.reset()
        self._writer.set_step(self._steps_done + self._len_epoch - 1, part)

        first = True
        for _, batch in tqdm(
            enumerate(dataloader),
            desc=part,
            total=len(dataloader),
        ):
            batch = self._process_batch(
                batch,
                is_train=False,
                metrics=self._evaluation_metrics,
            )
            if first:
                self._log_predictions(**batch)
                first = False

        self._writer.add_scalars(self._evaluation_metrics.result())

        return self._evaluation_metrics.result()

    def _log_predictions(
        self,
        *,
        text: list[str],
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
        audio_path: list[str],
        examples_to_log: int = 10,
        **kwargs: Any,
    ):
        examples_to_log = min(examples_to_log, log_probs.shape[0])

        argmax_inds = log_probs.detach().cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self._text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self._text_encoder.ctc_decode(inds) for inds in argmax_inds]

        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))

        # log examples_to_log exapmles to a table
        table_to_log = {}
        for i in range(examples_to_log):
            pred, target, raw_pred, current_path = tuples[i]
            target = BaseTextEncoder.normalize_text(target)
            table_to_log[i] = {
                "target": target,
                "predictions argmax": pred,
                "raw prediction argmax": raw_pred,
                "wer argmax": calc_wer(target, pred) * 100,
                "cer argmax": calc_cer(target, pred) * 100,
                "audio": current_path,
            }
        df = pd.DataFrame.from_dict(table_to_log, orient="index")
        df["raw prediction argmax"] = df["raw prediction argmax"].str.wrap(50)
        self._writer.add_text(
            "predictions",
            df.to_html(float_format="{:.2f}".format).replace("\\n", "<br>"),
        )

    def _log_train_spectrograms(
        self,
        *,
        audio_path: list[str],
        spectrogram: torch.Tensor,
        spectrogram_length: torch.Tensor,
        audio: torch.Tensor,
        examples_to_log: int = 6,
        **kwargs: Any,
    ):
        train_dataset = self._dataset_and_params["train"][0]
        specs_original: list[torch.Tensor] = []
        specs_processed: list[torch.Tensor] = []
        examples_to_log = min(examples_to_log, spectrogram.shape[0])

        for i in range(examples_to_log):
            spec_processed_len = spectrogram_length[i]
            spec_processed = spectrogram[i, :, :spec_processed_len].detach().cpu()

            audio_original, spec_original = train_dataset._process_wave(
                train_dataset._load_audio(audio_path[i])
            )

            self._writer.add_audio(
                f"train_audio/audio_original_{i}",
                audio_original,
                sample_rate=self._config["preprocessing"]["sr"],
                caption=audio_path[i],
            )
            self._writer.add_audio(
                f"train_audio/audio_processed_{i}",
                audio[i],
                sample_rate=self._config["preprocessing"]["sr"],
                caption=audio_path[i],
            )

            specs_original.append(spec_original.squeeze())
            specs_processed.append(spec_processed)

        with plot_spectrograms_to_buf(
            specs_original,
            specs_processed,
            audio_path[:examples_to_log],
        ) as buf:
            with PIL.Image.open(buf) as image:
                self._writer.add_image(
                    "train_audio/spectrograms",
                    image,
                )

    @torch.no_grad()
    def _log_model_weights(self):
        for name, p in self._model.named_parameters():
            self._writer.add_histogram(
                self._model.classify_parameter_name("weights", name),
                p,
                bins="auto",
            )

    @torch.no_grad()
    def _log_model_gradients(self):
        for name, p in self._model.named_parameters():
            if p.grad is not None:
                self._writer.add_histogram(
                    self._model.classify_parameter_name("gradients", name),
                    p,
                    bins="auto",
                )
            else:
                # assuming we train the whole model
                self._logger.warning(f"p.grad is None: {name}")
