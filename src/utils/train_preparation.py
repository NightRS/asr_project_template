import random

import numpy as np
import torch

import src.loss as module_loss
import src.metric as module_metric
import src.model as module_arch
from src.base.base_dataset import BaseDataset
from src.base.base_metric import BaseMetric
from src.base.base_model import BaseModel
from src.base.base_text_encoder import BaseCTCTextEncoder
from src.utils.object_loading import DataloaderParams, get_datasets
from src.utils.parse_config import ConfigParser

from .util import prepare_device


def fix_seed(seed: int):
    # maybe use PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG=:4096:8 as well?
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def prepare_train_components(
    config: ConfigParser, log_model_structure: bool = True
) -> tuple[
    BaseModel | torch.nn.DataParallel[BaseModel],
    torch.nn.Module,
    list[BaseMetric],
    torch.optim.Optimizer,
    torch.device,
    torch.optim.lr_scheduler.LRScheduler,
    dict[str, tuple[BaseDataset, DataloaderParams]],
    BaseCTCTextEncoder,
]:
    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataset_and_params = get_datasets(config, text_encoder)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch, n_class=len(text_encoder))
    if log_model_structure:
        logger = config.get_logger("train")
        logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        config["lr_scheduler"], torch.optim.lr_scheduler, optimizer
    )

    return (
        model,
        loss_module,
        metrics,
        optimizer,
        device,
        lr_scheduler,
        dataset_and_params,
        text_encoder,
    )
