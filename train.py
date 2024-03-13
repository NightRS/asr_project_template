import argparse
from pathlib import Path, PurePath

from src.logger.cloud_saver import CloudSaverBackendType
from src.trainer import Trainer
from src.utils.parse_config import ConfigParser, CustomArg
from src.utils.train_preparation import fix_seed, prepare_train_components


def main(config: ConfigParser):
    fix_seed(123)

    (
        model,
        loss_module,
        metrics,
        optimizer,
        device,
        lr_scheduler,
        dataset_and_params,
        text_encoder,
    ) = prepare_train_components(config)

    trainer = Trainer(
        model=model,
        criterion=loss_module,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        lr_scheduler=lr_scheduler,
        dataset_and_params=dataset_and_params,
        text_encoder=text_encoder,
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        type=Path,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="run name",
    )
    args.add_argument(
        "-r",
        "--resume",
        type=Path,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-rc",
        "--resume_cloud",
        type=PurePath,
        help="path in the cloud to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-ct",
        "--cloud_type",
        type=str,
        choices=[cloud.name for cloud in CloudSaverBackendType],
        help="cloud backend type; required if --resume_cloud is used",
    )
    args.add_argument(
        "-d",
        "--device",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    options = [
        CustomArg(("-lr", "--learning_rate"), type=float, target="optimizer;args;lr"),
        CustomArg(("-e", "--epochs"), type=int, target="trainer;epochs"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
