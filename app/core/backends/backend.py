from enum import Enum

from abc import ABC, abstractmethod

from core.model.ModelType import ModelType
from core.model.models_constants import RK_TAGS_LIST


class BackendType(Enum):
    pass


class BackendException(Exception):
    pass


class BackendType(str, Enum):
    RKLLM = "RKLLM"
    RKNN = "RKNN"

    @classmethod
    def from_model_type(cls, model_type: ModelType) -> BackendType:
        for value in BackendType:
            if value.value == model_type.value:
                return value
        raise BackendException(f"ModelType {model_type} not found in BackendType")


BACKEND_SUPPORTED_LIB_VERSION = {
    BackendType.RKLLM: ["1.0.0"],
    BackendType.RKNN: ["1.0.0"],
}

# validation process
for backend_type in BackendType:
    assert (
        backend_type.value.lower() in RK_TAGS_LIST
    ), f"Missing {backend_type.value.lower()} in RK_TAGS_LIST"
    assert (
        backend_type in BACKEND_SUPPORTED_LIB_VERSION.keys()
    ), f"Missing {backend_type} in BACKEND_SUPPORTED_LIB_VERSION"
    assert (
        ModelType(backend_type.value) is not None
    ), f"Missing {backend_type.value} in ModelType"


class Backend(ABC):
    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type

    @abstractmethod
    def run(self, param):
        pass

    @abstractmethod
    def abort(self):
        pass

    @abstractmethod
    def clear_cache(self):
        pass

    @abstractmethod
    def release(self):
        pass
