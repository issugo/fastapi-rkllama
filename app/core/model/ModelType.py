from enum import Enum


class ModelType(str, Enum):
    RKLLM = "RKLLM"
    RKNN = "RKNN"

    def get_extension(self):
        if self == ModelType.RKLLM:
            return ".rkllm"
        elif self == ModelType.RKNN:
            return ".rknn"
