from enum import Enum

from pydantic import BaseModel

from core.model.ModelType import ModelType
from core.model.models_constants import RK_TAGS_LIST


class BackendType(str, Enum):
    RKLLM = "RKLLM"
    RKNN = "RKNN"


BACKEND_SUPPORTED_LIB_VERSION = {
    BackendType.RKLLM: ["1.0.0"],
    BackendType.RKNN: ["1.0.0"],
}

# validation process
for backend_type in BackendType:
    assert backend_type.value.lower() in RK_TAGS_LIST, f"Missing {backend_type.value.lower()} in RK_TAGS_LIST"
    assert backend_type in BACKEND_SUPPORTED_LIB_VERSION.keys(), f"Missing {backend_type} in BACKEND_SUPPORTED_LIB_VERSION"
    assert ModelType(backend_type.value) is not None, f"Missing {backend_type.value} in ModelType"


class Backend(BaseModel):
    backend_type: BackendType
