import importlib
import io
import json
import logging
import os
from argparse import ArgumentParser
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial, reduce
from operator import getitem
from pathlib import Path, PurePath
from types import ModuleType
from typing import Any, Optional

from src import text_encoder as text_encoder_module
from src.base.base_text_encoder import BaseCTCTextEncoder
from src.logger.cloud_saver import BaseCloudSaver, get_cloud_saver
from src.logger.logger import setup_logging
from src.text_encoder import CTCCharTextEncoder
from src.utils.util import ROOT_PATH, read_json, write_json


@dataclass
class CustomArg:
    flags: Iterable[str]
    type: type[Any]
    target: str


@dataclass
class ResumeSettings:
    pass


@dataclass
class ResumeLocal(ResumeSettings):
    resume_path: Path


@dataclass
class ResumeCloud(ResumeSettings):
    resume_path: PurePath
    cloud_saver: BaseCloudSaver

    def __post_init__(self):
        if not self.resume_path.is_absolute():
            raise ValueError("resume_path must be absolute for cloud paths")


class ConfigParser:
    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        resume: Optional[ResumeSettings] = None,
        modification: Optional[dict[str, Any]] = None,
    ):
        """
        Class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param run_name: Unique Identifier for training processes.
                         Used to save checkpoints and training log.
        :param config: Dict containing configurations, hyperparameters for training.
                       contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict {keychain: value}, specifying position values to be replaced
                             from config dict.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self._resume = resume
        self._text_encoder = None

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        self._run_name = run_name
        self._save_dir = save_dir / "models" / exper_name / run_name
        self._log_dir = save_dir / "log" / exper_name / run_name
        self._cloud_dir = (
            PurePath(self.config["trainer"]["cloud_dir"]) / exper_name / run_name
        )

        assert self.cloud_dir.is_absolute(), "cloud_dir path must be absolute"

        # make directory for saving checkpoints and log.
        exist_ok = run_name == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / "config.json")

        # configure logging module
        setup_logging(self.log_dir)
        self._log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, raw_args: ArgumentParser, options: list[CustomArg]):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            raw_args.add_argument(*opt.flags, type=opt.type)

        args = raw_args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume or args.resume_cloud:
            if args.resume_cloud:
                assert (
                    args.cloud_type is not None
                ), "You need to specify --cloud_type if --resume_cloud is used."

                resume = ResumeCloud(
                    resume_path=args.resume_cloud,
                    cloud_saver=get_cloud_saver(args.cloud_type),
                )

                buf = io.BytesIO()
                resume.cloud_saver.download(resume.resume_path / "config.json", buf)
                buf.seek(0)

                config = json.loads(buf.read())

            else:
                resume = ResumeLocal(
                    resume_path=args.resume,
                )
                config = read_json(resume.resume_path)
        else:
            assert args.config is not None, (
                "Configuration file need to be specified. "
                "Add '-c config.json', for example."
            )
            resume = None
            config = read_json(args.config)

        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }
        return cls(
            run_name=args.run_name,
            config=config,
            resume=resume,
            modification=modification,
        )

    @staticmethod
    def init_obj(
        obj_dict: dict[str, Any],
        default_module: ModuleType,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj(config['param'], module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if "module" in obj_dict:
            default_module = importlib.import_module(obj_dict["module"])

        module_name = obj_dict["type"]
        module_args = dict(obj_dict["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(default_module, module_name)(*args, **module_args)

    def init_ftn(
        self,
        name: str,
        module: ModuleType,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name: str) -> Any:
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name: str | None, verbosity: int = 2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self._log_levels.keys()
        )
        assert verbosity in self._log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self._log_levels[verbosity])
        return logger

    def get_text_encoder(self) -> BaseCTCTextEncoder:
        if self._text_encoder is None:
            if "text_encoder" not in self._config:
                self._text_encoder = CTCCharTextEncoder()
            else:
                self._text_encoder = self.init_obj(
                    self["text_encoder"], default_module=text_encoder_module
                )
        return self._text_encoder

    # setting read-only attributes
    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def cloud_dir(self) -> PurePath:
        return self._cloud_dir

    @property
    def resume(self) -> Optional[ResumeSettings]:
        return self._resume

    @classmethod
    def get_test_configs(cls, run_name: str, **kwargs: Any):
        config_path = ROOT_PATH / "src" / "tests" / "config.json"
        return cls(run_name=run_name, config=read_json(config_path), **kwargs)


# helper functions to update config dict with custom cli options
def _update_config(config: dict[str, Any], modification: Optional[dict[str, Any]]):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags: Iterable[str]) -> str:
    for flag in flags:
        return flag.lstrip("-")
    raise ValueError(f"Can't infer option name from flags: {flags}")


def _set_by_path(tree: dict[str, Any], keys: str, value: Any):
    """Set a value in a nested object in tree by sequence of keys."""
    keys_list = keys.split(";")
    _get_by_path(tree, keys_list[:-1])[keys_list[-1]] = value


def _get_by_path(tree: dict[str, Any], keys: list[str]):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
