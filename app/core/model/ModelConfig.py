from typing import get_type_hints, Any

from pydantic import BaseModel

from core.config.DefaultModelConfig import DefaultModelConfig
from core.model.ModelMetadata import SimpleModelMetadata
from core.model.ModelPath import ModelPath


class MinimalModelConfig(BaseModel):
    FROM: str
    HUGGINGFACE_PATH: str
    SYSTEM: str

    def dump(self, backend_model_file: str):
        with open(backend_model_file, "w") as f:
            for attr in self.__dict__:
                f.write(f"{attr.upper()}={self.__dict__[attr]}")

class MinimalTemperedModelConfig(MinimalModelConfig):
    temperature: float

class ModelConfig(MinimalTemperedModelConfig):
    enable_thinking: bool
    num_ctx: int
    max_new_tokens: int
    top_k: int
    top_p: float
    repeat_penalty: float
    frequency_penalty: float
    presence_penalty: float
    mirostat: bool
    mirostat_tau: float
    mirostat_eta: float

    @classmethod
    def _infer_and_convert_type(cls, key: str, value: str) -> Any:
        """
        Converts string values to appropriate Python types.

        Uses schema if available, otherwise applies heuristic type detection
        for booleans, numbers, and lists.
        """
        # Handle None values
        if value is None:
            return None

        # Check if we already know the type from schema
        field_type = get_type_hints(cls).get(key)
        if field_type is not None:
            match field_type:
                case bool():
                    return value.lower() in ["1", "true", "yes", "on"]
                case int():
                    return int(value)
                case float():
                    return float(value)
                case _:
                    return value
        return None


    @classmethod
    def create(cls, model_path: ModelPath, model_metadata: SimpleModelMetadata, default_model_config: DefaultModelConfig):
        if model_metadata.model_type is not None:
            if model_path.model_type != model_metadata.model_type:
                raise ValueError(f"Model type mismatch: {model_path.model_type} != {model_metadata.model_type}")

        data: dict = model_path.__dict__

        if "endpoint_model_file" in data:
            data["FROM"] = data["endpoint_model_file"]
            del data["endpoint_model_file"]
            from_ext = data["FROM"].split(".")[-1]
            if f".{from_ext}" != model_path.model_type.get_extension():
                raise ValueError(f"Model type mismatch: FROM extension {from_ext} != {model_path.model_type}")

        if "huggingface_path" in data:
            data["HUGGINGFACE_PATH"] = data["huggingface_path"]
            del data["huggingface_path"]

        data.update(model_metadata.__dict__)

        if "system_prompt" in data:
            data["SYSTEM"] = data["system_prompt"]
            del data["system_prompt"]

        default_data = {}
        for attr in default_model_config.__dict__:
            default_data[attr.removeprefix("default_")] = default_model_config.__dict__[attr]
        data.update(default_data)

        return cls(**data)

    @classmethod
    def load(cls, model_path: ModelPath):
        data = {}
        with open(model_path.modelfile, "r") as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                key, value = line.strip().split("=")
                if key.strip() in ["TEMPERATURE"]:
                    data[key.strip().lower()] = float(value)
                elif key.strip().lower() in ["from", "huggingface_path", "system"]:
                    data[key.strip().upper()] = cls._infer_and_convert_type(key.strip().upper(), value)
                else:
                    data[key.strip().lower()] = cls._infer_and_convert_type(key.strip().lower(), value)
        return cls(**data)
