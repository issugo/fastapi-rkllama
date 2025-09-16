from enum import Enum


class ModelType(str, Enum):
    RKLLM = "RKLLM"
    RKNN = "RKNN"

    def get_extension(self):
        if self == ModelType.RKLLM:
            return ".rkllm"
        elif self == ModelType.RKNN:
            return ".rknn"

    @classmethod
    def get_model_type_from_endpoint_model_file(cls, endpoint_model_file: str):
        for mtype in cls:
            if endpoint_model_file.endswith(mtype.get_extension()):
                return mtype
        return None