import asyncio
import logging
import os
from pathlib import Path, PurePath
from typing import BinaryIO

import yadisk

from src.base.base_cloud import BaseCloudSaver

logger = logging.getLogger(__name__)


class YaDiskSaver(BaseCloudSaver):
    TIMEOUT = 300.0

    def __init__(self):
        self._token = os.environ["YADISK_TOKEN"]
        assert self._token != "", (
            "To use YaDiskSaver, you should provide token as a "
            "YADISK_TOKEN environment variable"
        )

        with yadisk.Client(token=self._token) as client:
            logger.info("YaDisk token is valid:", client.check_token())

    def prepare_log_folder(self, path: PurePath):
        for parent_path in path.parents[-2::-1]:
            self.mkdir(parent_path)

    def download(self, path_from: PurePath, path_to: Path | BinaryIO):
        try:
            with yadisk.Client(token=self._token) as client:
                if isinstance(path_to, Path):
                    client_path_to = str(path_to)
                else:
                    client_path_to = path_to
                client.download(str(path_from), client_path_to, timeout=self.TIMEOUT)
        finally:
            logger.info(f"Download call ended from {path_from} to {path_to}")

    def mkdir(self, path: PurePath, exist_ok: bool = True):
        try:
            with yadisk.Client(token=self._token) as client:
                try:
                    client.mkdir(str(path))
                except yadisk.exceptions.PathExistsError:
                    if exist_ok:
                        pass
                    else:
                        raise
        except Exception:
            logger.warning(f"Unsuccessful mkdir, path: {path}", exc_info=True)
        finally:
            logger.info(f"mkdir call ended; path: {path}")

    def upload(self, path_from: Path, path_to: PurePath, overwrite: bool = False):
        self.prepare_log_folder(path_to)
        try:
            with yadisk.Client(token=self._token) as client:
                client.upload(
                    str(path_from),
                    str(path_to),
                    overwrite=overwrite,
                    timeout=self.TIMEOUT,
                )
        except Exception:
            logger.warning(
                f"Unsuccessful upload from {path_from} to {path_to}", exc_info=True
            )
        finally:
            logger.info(f"Upload call ended from {path_from} to {path_to}")

    def ensure_files_upload(self):
        return


class AsyncYaDiskSaver(YaDiskSaver):
    TIMEOUT = 300.0

    def __init__(self):
        super().__init__()
        self._task_queue: list[asyncio.Task[None]] = []

    def upload(self, path_from: Path, path_to: PurePath, overwrite: bool = False):
        async def save_impl():
            self.prepare_log_folder(path_to)
            try:
                async with yadisk.AsyncClient(token=self._token) as client:
                    await client.upload(
                        str(path_from),
                        str(path_to),
                        overwrite=overwrite,
                        timeout=self.TIMEOUT,
                    )
            except Exception:
                logger.warning(
                    f"Unsuccessful upload from {path_from} to {path_to}", exc_info=True
                )
            finally:
                logger.info(
                    f"Upload call ended from {path_from} to {path_to}. "
                    f"It may take a while to complete."
                )

        self._task_queue.append(asyncio.create_task(save_impl()))

    def ensure_files_upload(self):
        logger.info(f"Waiting for uploads from previous save ...")
        asyncio.gather(*self._task_queue)
        logger.info(f"No more uploads in queue.")
        self._task_queue = []
