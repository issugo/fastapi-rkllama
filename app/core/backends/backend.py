from enum import Enum
from typing import Any

from pydantic import BaseModel


class BackendType(str, Enum):
    RKLLM = "RKLLM"
    RKNN = "RKNN"

class Backend(BaseModel):
    backend_type: BackendType
