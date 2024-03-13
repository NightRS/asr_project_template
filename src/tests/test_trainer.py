import math
import os
import shutil
import unittest
from datetime import datetime
from itertools import product
from pathlib import Path, PurePath
from typing import Any, Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.base.base_dataset import BaseDataset
from src.base.base_metric import BaseMetric
from src.base.base_model import BaseModel
from src.base.base_text_encoder import BaseTextEncoder
from src.logger.wandb import WanDBWriter
from src.logger.yadisk import AsyncYaDiskSaver
from src.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from src.trainer import Trainer
from src.utils.object_loading import DataloaderParams, get_dataloader_from_params
from src.utils.parse_config import ConfigParser, ResumeCloud, ResumeLocal
from src.utils.train_preparation import fix_seed, prepare_train_components
from src.utils.util import ROOT_PATH, loop_exactly_n_times

from .check_writer_log import check1, check5, check12
from .utils import clear_log_folder_after_use, models_to_test_config


class DebugWanDBWriter(WanDBWriter):
    debug_report: list[tuple[int, Any]] = []

    def log_manual(self, data: Any, step: int):
        DebugWanDBWriter.debug_report.append((step, data))
        return super().log_manual(data, step)

    def write(self):
        DebugWanDBWriter.debug_report.append((self._step, self._log))
        return super().write()


class DebugDatetime(datetime):
    i = 0

    @classmethod
    def now(cls, tz: Any = None):
        answer = cls.fromtimestamp(
            0.9 * cls.i + math.sin(0.7 * cls.i) + 1.06**cls.i + cls.i**0.36
        )
        cls.i += 1
        return answer


class DebugTrainer(Trainer):
    debug_report: list[dict[str, Any]] = []

    def _train_epoch(self, epoch: int):
        res = super()._train_epoch(epoch)
        DebugTrainer.debug_report.append(res)
        return res


class DummyModel(BaseModel):
    def __init__(self, n_feats: int, n_class: int, fc_hidden: int = 512, **kwargs: Any):
        super().__init__(n_feats, n_class, **kwargs)
        self.net = torch.nn.Linear(n_feats, n_class)
        self.net.weight.requires_grad = False
        self.net.bias.requires_grad = False
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.grads = torch.abs(
            torch.randn(12 * 3, generator=torch.Generator().manual_seed(11))
        )
        self.batch_i = 0
        self.grad_i = 0

    def forward(self, *, spectrogram: torch.Tensor, **batch: Any) -> dict[str, Any]:
        if self.training:
            self.p.grad = torch.ones(1) * self.grads[self.grad_i]
            self.grad_i += 1
        answer = {"logits": self.net(spectrogram.transpose(1, 2)), "i": self.batch_i}
        self.batch_i += 1
        return answer

    def transform_input_lengths(self, input_lengths: int) -> int:
        return input_lengths

    def classify_parameter_name(self, prefix: str, name: str) -> str:
        return f"{prefix}/{name}"


class DummyLoss(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        self.losses = torch.randn(
            (12 + 4) * 3,
            generator=torch.Generator().manual_seed(22),
        )

    def forward(self, **batch: Any) -> torch.Tensor:
        return torch.ones(1, requires_grad=True) * self.losses[batch["i"]]


class DummyMetric(BaseMetric):
    def __init__(
        self,
        *,
        text_encoder: BaseTextEncoder = CTCCharTextEncoder(),
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.metrics = torch.randn(
            (12 + 4) * 3,
            generator=torch.Generator().manual_seed(33),
        )

    def __call__(self, **batch: Any) -> float:
        return (torch.ones(1) * self.metrics[batch["i"]]).item()


def get_lrs(config: ConfigParser) -> list[float]:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    lr_scheduler = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer
    )

    lrs: list[float] = []
    for _ in range(12 * 3):
        optimizer.step()
        lr_scheduler.step()
        lrs.append(lr_scheduler.get_last_lr()[0])
    return lrs


def manual_training(
    model: BaseModel | torch.nn.DataParallel[BaseModel],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataset_and_params: dict[str, tuple[BaseDataset, DataloaderParams]],
    epochs: int,
    len_epoch: int,
):
    train_dataloader = get_dataloader_from_params(*dataset_and_params["train"])

    for epoch in tqdm(range(epochs)):
        dataloader = loop_exactly_n_times(train_dataloader, len_epoch)
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            output = model(**batch)

            if isinstance(output, dict):
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )

            loss = criterion(**batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()


class TestTrainer(unittest.TestCase):
    def setUp(self):
        def float_compare(a: float, b: float, msg: Optional[str] = None):
            to_show = (
                f"{a} != {b}, rel_tol = 1e-9, diff = {abs(a-b)}, "
                f"rel_tol * max(abs(a), abs(b)) = {1e-9 * max(abs(a), abs(b))}"
            )
            if msg is not None:
                to_show = f"{to_show} : {msg}"
            assert math.isclose(a, b, rel_tol=1e-9), to_show

        self.addTypeEqualityFunc(float, float_compare)

    def tearDown(self):
        paths = [
            ROOT_PATH / "saved" / "log",
            ROOT_PATH / "saved" / "models",
        ]
        for path in paths:
            for d in os.listdir(path):
                if d.startswith("test_"):
                    shutil.rmtree(path / d)
        if "RUN_ID_FOR_WANDB" in os.environ:
            os.environ.pop("RUN_ID_FOR_WANDB")

    def test_core_training_logic_and_reproducibility(self):
        for device_trainer, device_manual, model in product(
            ["cpu", "gpu"],
            ["cpu", "gpu"],
            models_to_test_config(),
        ):
            with self.subTest(
                device_trainer=device_trainer,
                device_manual=device_manual,
                model=model["type"],
            ):
                if not torch.cuda.is_available() and (
                    any(device == "gpu" for device in [device_trainer, device_manual])
                ):
                    self.skipTest("CUDA not available")

                config_trainer = ConfigParser.get_test_configs(
                    run_name="test_core_trainer",
                    modification={
                        "n_gpu": 0 if device_trainer == "cpu" else 1,
                        "arch": model,
                    },
                )
                config_manual = ConfigParser.get_test_configs(
                    run_name="test_core_manual",
                    modification={
                        "n_gpu": 0 if device_manual == "cpu" else 1,
                        "arch": model,
                    },
                )
                with clear_log_folder_after_use(config_trainer, config_manual):
                    fix_seed(123)
                    (
                        model_trainer,
                        loss_module_trainer,
                        metrics_trainer,
                        optimizer_trainer,
                        torch_device_trainer,
                        lr_scheduler_trainer,
                        dataset_and_params_trainer,
                        text_encoder_trainer,
                    ) = prepare_train_components(
                        config_trainer, log_model_structure=False
                    )

                    fix_seed(123)
                    (
                        model_manual,
                        loss_module_manual,
                        _,
                        optimizer_manual,
                        _,
                        lr_scheduler_manual,
                        dataset_and_params_manual,
                        _,
                    ) = prepare_train_components(
                        config_manual, log_model_structure=False
                    )

                    trainer = Trainer(
                        model=model_trainer,
                        criterion=loss_module_trainer,
                        metrics=metrics_trainer,
                        optimizer=optimizer_trainer,
                        config=config_trainer,
                        device=torch_device_trainer,
                        lr_scheduler=lr_scheduler_trainer,
                        dataset_and_params=dataset_and_params_trainer,
                        text_encoder=text_encoder_trainer,
                    )
                    trainer.train()
                    manual_training(
                        model=model_manual,
                        criterion=loss_module_manual,
                        optimizer=optimizer_manual,
                        lr_scheduler=lr_scheduler_manual,
                        dataset_and_params=dataset_and_params_manual,
                        epochs=config_manual["trainer"]["epochs"],
                        len_epoch=config_manual["trainer"]["len_epoch"],
                    )

                torch.testing.assert_close(
                    model_manual.state_dict(),
                    model_trainer.state_dict(),
                    check_device=False,
                )
                torch.testing.assert_close(
                    optimizer_manual.state_dict(),
                    optimizer_trainer.state_dict(),
                    check_device=False,
                )
                self.assertEqual(
                    lr_scheduler_manual.state_dict(),
                    lr_scheduler_trainer.state_dict(),
                )

    def test_val_batch_independence(self):
        for model in models_to_test_config():
            with self.subTest(model=model["type"]):
                config_1 = ConfigParser.get_test_configs(
                    run_name="test_batch_independence_1",
                    modification={"arch": model, "data;val;batch_size": 1},
                )
                config_16 = ConfigParser.get_test_configs(
                    run_name="test_batch_independence_16",
                    modification={"arch": model, "data;val;batch_size": 16},
                )
                with clear_log_folder_after_use(config_1, config_16):
                    results: list[list[dict[str, Any]]] = []
                    for config in [config_1, config_16]:
                        DebugTrainer.debug_report = []
                        fix_seed(123)
                        (
                            model,
                            loss_module,
                            metrics,
                            optimizer,
                            torch_device,
                            lr_scheduler,
                            dataset_and_params,
                            text_encoder,
                        ) = prepare_train_components(config, log_model_structure=False)
                        trainer = DebugTrainer(
                            model=model,
                            criterion=loss_module,
                            metrics=metrics,
                            optimizer=optimizer,
                            config=config,
                            device=torch_device,
                            lr_scheduler=lr_scheduler,
                            dataset_and_params=dataset_and_params,
                            text_encoder=text_encoder,
                        )
                        trainer.train()
                        results.append(DebugTrainer.debug_report)

                result1, result16 = results
                for i in range(len(result1)):
                    for key in result1[i]:
                        if key.startswith("val_"):
                            self.assertEqual(
                                result1[i][key],
                                result16[i][key],
                            )

    @patch("src.metric.ArgmaxCERMetric", DummyMetric)
    @patch("src.loss.CTCLoss", DummyLoss)
    @patch("src.model.BaselineModel", DummyModel)
    @patch("src.logger.visualization.WanDBWriter", DebugWanDBWriter)
    @patch("src.logger.wandb.datetime", DebugDatetime)
    def test_train_writer(self):
        for log_frequency, check_function in zip(
            [1, 5, 12, 99],
            [check1, check5, check12, check12],
        ):
            with self.subTest(log_frequency=log_frequency):
                DebugWanDBWriter.debug_report = []
                DebugDatetime.i = 0

                config = ConfigParser.get_test_configs(
                    run_name="test_train_writer_{i}",
                    modification={"trainer;log_step": log_frequency},
                )
                with clear_log_folder_after_use(config):
                    fix_seed(123)
                    (
                        model,
                        loss_module,
                        metrics,
                        optimizer,
                        torch_device,
                        lr_scheduler,
                        dataset_and_params,
                        text_encoder,
                    ) = prepare_train_components(config, log_model_structure=False)
                    trainer = Trainer(
                        model=model,
                        criterion=loss_module,
                        metrics=metrics,
                        optimizer=optimizer,
                        config=config,
                        device=torch_device,
                        lr_scheduler=lr_scheduler,
                        dataset_and_params=dataset_and_params,
                        text_encoder=text_encoder,
                    )
                    trainer.train()

                DebugDatetime.i = 0
                expected_output = check_function(
                    timestamps=[DebugDatetime.now() for _ in range(46)],
                    grads=DummyModel(1, 1).grads.tolist(),
                    losses=DummyLoss(0, False).losses.tolist(),
                    metrics=DummyMetric().metrics.tolist(),
                    lrs=get_lrs(config),
                )
                actual_output = DebugWanDBWriter.debug_report

                self.assertEqual(len(expected_output), len(actual_output))

                for i in range(len(expected_output)):
                    step_expected, data_expected = expected_output[i]
                    step_actual, data_actual = actual_output[i]

                    self.assertEqual(
                        step_expected,
                        step_actual,
                        f"{i}: wrong step, expected {step_expected}, actual {step_actual}",
                    )
                    self.assertEqual(
                        set(data_expected.keys()),
                        set(data_actual.keys()),
                        f"{i}: mismatch on step {step_expected}: wrong keys",
                    )
                    for key in data_expected.keys():
                        self.assertEqual(
                            data_expected[key],
                            data_actual[key],
                            f"{i}: mismatch on step {step_expected}, key={key}",
                        )

    def test_train_resume(self):
        for n_exp, (
            device_continious,
            device_1,
            device_2,
            device_3,
            resume2,
            resume3,
            model,
        ) in enumerate(
            product(
                ["cpu"],
                ["cpu", "gpu"],
                ["cpu"],
                ["cpu", "gpu"],
                ["cloud"],
                ["local"],
                models_to_test_config(),
            )
        ):
            with self.subTest(
                device_continious=device_continious,
                device_1=device_1,
                device_2=device_2,
                device_3=device_3,
                resume2=resume2,
                resume3=resume3,
                model=model["type"],
            ):
                if not torch.cuda.is_available() and (
                    any(device == "gpu" for device in [device_1, device_2, device_3])
                ):
                    self.skipTest("CUDA not available")

                resume = [
                    None,
                    (
                        ResumeCloud(
                            resume_path=(
                                PurePath("/test_asr")
                                / "test_config"
                                / f"{n_exp}_test_1"
                                / "checkpoint_epoch_1"
                            ),
                            cloud_saver=AsyncYaDiskSaver(),
                        )
                        if resume2 == "cloud"
                        else ResumeLocal(
                            resume_path=(
                                Path("saved/models")
                                / "test_config"
                                / f"{n_exp}_test_1"
                                / "checkpoint_epoch_1"
                            )
                        )
                    ),
                    (
                        ResumeCloud(
                            resume_path=(
                                PurePath("/test_asr")
                                / "test_config"
                                / f"{n_exp}_test_2"
                                / "checkpoint_epoch_2"
                            ),
                            cloud_saver=AsyncYaDiskSaver(),
                        )
                        if resume3 == "cloud"
                        else ResumeLocal(
                            resume_path=(
                                Path("saved/models")
                                / "test_config"
                                / f"{n_exp}_test_2"
                                / "checkpoint_epoch_2"
                            )
                        )
                    ),
                ]

                config_continious = ConfigParser.get_test_configs(
                    run_name="test_continious",
                    modification={
                        "n_gpu": 0 if device_continious == "cpu" else 1,
                        "arch": model,
                    },
                )
                with clear_log_folder_after_use(config_continious):
                    fix_seed(123)
                    (
                        model_continious,
                        loss_module_continious,
                        metrics_continious,
                        optimizer_continious,
                        torch_device_continious,
                        lr_scheduler_continious,
                        dataset_and_params_continious,
                        text_encoder_continious,
                    ) = prepare_train_components(
                        config_continious, log_model_structure=False
                    )
                    trainer_continious = Trainer(
                        model=model_continious,
                        criterion=loss_module_continious,
                        metrics=metrics_continious,
                        optimizer=optimizer_continious,
                        config=config_continious,
                        device=torch_device_continious,
                        lr_scheduler=lr_scheduler_continious,
                        dataset_and_params=dataset_and_params_continious,
                        text_encoder=text_encoder_continious,
                    )
                    trainer_continious.train()

                model_i = None
                optimizer_i = None
                lr_scheduler_i = None
                configs: list[ConfigParser] = []

                run_id = None

                for i, device_i in zip(
                    range(1, 4),
                    [device_1, device_2, device_3],
                ):
                    if i > 1:
                        assert run_id is not None
                        os.environ["RUN_ID_FOR_WANDB"] = run_id

                    config_i = ConfigParser.get_test_configs(
                        run_name=f"{n_exp}_test_{i}",
                        modification={
                            "n_gpu": 0 if device_i == "cpu" else 1,
                            "arch": model,
                            "trainer;epochs": i,
                            "trainer;cloud_saver": "yadisk",
                        },
                        resume=resume[i - 1],
                    )
                    configs.append(config_i)
                    fix_seed(123)
                    (
                        model_i,
                        loss_module_i,
                        metrics_i,
                        optimizer_i,
                        torch_device_i,
                        lr_scheduler_i,
                        dataset_and_params_i,
                        text_encoder_i,
                    ) = prepare_train_components(config_i, log_model_structure=False)
                    trainer_i = Trainer(
                        model=model_i,
                        criterion=loss_module_i,
                        metrics=metrics_i,
                        optimizer=optimizer_i,
                        config=config_i,
                        device=torch_device_i,
                        lr_scheduler=lr_scheduler_i,
                        dataset_and_params=dataset_and_params_i,
                        text_encoder=text_encoder_i,
                    )
                    trainer_i.train()
                    if i == 1:
                        run_id = trainer_i._writer._run_id

                with clear_log_folder_after_use(*configs):
                    pass

                assert model_i is not None
                assert optimizer_i is not None
                assert lr_scheduler_i is not None
                torch.testing.assert_close(
                    model_continious.state_dict(),
                    model_i.state_dict(),
                    check_device=False,
                )
                torch.testing.assert_close(
                    optimizer_continious.state_dict(),
                    optimizer_i.state_dict(),
                    check_device=False,
                )
                self.assertEqual(
                    lr_scheduler_continious.state_dict(),
                    lr_scheduler_i.state_dict(),
                )
