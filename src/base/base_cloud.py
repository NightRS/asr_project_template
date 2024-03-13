from pathlib import Path, PurePath
from typing import BinaryIO


class BaseCloudSaver:
    def prepare_log_folder(self, path: PurePath):
        pass

    def download(self, path_from: PurePath, path_to: Path | BinaryIO):
        pass

    def mkdir(self, path: PurePath, exist_ok: bool = True):
        pass

    def upload(self, path_from: Path, path_to: PurePath, overwrite: bool = False):
        pass

    def ensure_files_upload(self):
        pass
