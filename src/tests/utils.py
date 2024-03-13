import shutil
from contextlib import contextmanager

from src.model import BaselineModel, DeepSpeech2
from src.utils.parse_config import ConfigParser


@contextmanager
def clear_log_folder_after_use(*config_parsers: ConfigParser):
    # this context manager deletes the log folders weather the body was executed succesfully or not
    try:
        yield config_parsers
    finally:
        for c in config_parsers:
            shutil.rmtree(c.save_dir)
            shutil.rmtree(c.log_dir)


def models_to_test_config():
    return [
        {"type": "BaselineModel", "args": {"n_feats": 64, "fc_hidden": 96}},
        {"type": "DeepSpeech2", "args": {"n_feats": 64, "rnn_hidden": 128}},
    ]


def models_to_test_instances(n_mels: int, alphabet_size: int):
    return [
        BaselineModel(n_feats=n_mels, n_class=alphabet_size),
        DeepSpeech2(n_feats=n_mels, n_class=alphabet_size, rnn_hidden=128),
    ]
