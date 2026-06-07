from enum import Enum

MODELTYPE_RKLLM: str = "RKLLM"
MODELTYPE_RKNN: str = "RKNN"
MODELTYPE_GGUF: str = "GGUF"


class ModelType(str, Enum):
    RKLLM = MODELTYPE_RKLLM
    RKNN = MODELTYPE_RKNN
    GGUF = MODELTYPE_GGUF

    def get_extension(self):
        if self == ModelType.RKLLM:
            return ".rkllm"
        elif self == ModelType.RKNN:
            return ".rknn"
        elif self == ModelType.GGUF:
            return ".gguf"

    @classmethod
    def get_model_type_from_endpoint_model_file(cls, endpoint_model_file: str):
        for mtype in cls:
            if endpoint_model_file.endswith(mtype.get_extension()):
                return mtype
        return None


FILE_TYPE = {MODELTYPE_RKLLM: 15, MODELTYPE_RKNN: -1, MODELTYPE_GGUF: -1}

# validation process
for model_type in ModelType:
    assert model_type.value in FILE_TYPE, f"Missing {model_type} in FILE_TYPE"

for model_type in FILE_TYPE:
    assert ModelType(model_type) is not None, f"Missing {model_type} in ModelType"
    assert (
        ModelType(model_type).get_extension() is not None
    ), f"Missing {model_type} extension in ModelType"
