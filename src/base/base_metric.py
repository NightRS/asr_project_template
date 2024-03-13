from typing import Any, Optional


class BaseMetric:
    def __init__(self, *, name: Optional[str] = None, **kwargs: Any):
        self.name = name if name is not None else type(self).__name__

    def __call__(self, **kwargs: Any) -> float:
        raise NotImplementedError()
