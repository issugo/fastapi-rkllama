from enum import Enum

class BackendType(str, Enum):
    RKLLM = "RKLLM"
    RKNN = "RKNN"

class Backend:
    def __init__(self, backend_type: BackendType):
        self._backend_type = backend_type

    @property
    def backend_type(self) -> BackendType:
        return self._backend_type