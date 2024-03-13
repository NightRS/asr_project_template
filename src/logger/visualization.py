from enum import Enum

from src.utils.parse_config import ConfigParser

from .wandb import WanDBWriter


class VisualizerBackendType(str, Enum):
    wandb = "wandb"


def get_visualizer(config: ConfigParser, backend: VisualizerBackendType) -> WanDBWriter:
    if backend == VisualizerBackendType.wandb:
        return WanDBWriter(config)

    raise ValueError("only wandb writer is currently supported")
