from enum import Enum

from src.base.base_cloud import BaseCloudSaver

from .yadisk import AsyncYaDiskSaver, YaDiskSaver


class CloudSaverBackendType(str, Enum):
    none = "none"
    yadisk = "yadisk"
    async_yadisk = "async_yadisk"


def get_cloud_saver(backend: CloudSaverBackendType) -> BaseCloudSaver:
    if backend == CloudSaverBackendType.none:
        return BaseCloudSaver()
    elif backend == CloudSaverBackendType.yadisk:
        return YaDiskSaver()
    elif backend == CloudSaverBackendType.async_yadisk:
        return AsyncYaDiskSaver()
    else:
        raise ValueError("unknown cloud saver type")
