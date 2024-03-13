import logging
import logging.config
from pathlib import Path
from typing import Optional

from src.utils.util import ROOT_PATH, read_json


def setup_logging(
    save_dir: Path,
    log_config_path: Optional[Path] = None,
    default_level: int | str = logging.INFO,
):
    """
    Setup logging configuration
    """
    if log_config_path is None:
        log_config_path = ROOT_PATH / "src" / "logger" / "logger_config.json"
    if log_config_path.is_file():
        config = read_json(log_config_path)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(
                log_config_path
            )
        )
        logging.basicConfig(level=default_level)
