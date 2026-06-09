import json
import re
from pathlib import Path
from typing import get_type_hints, Any, Optional

from pydantic import Field, BaseModel

from core.config.DefaultModelConfig import DefaultConfig
from core.config.config_utils import get_settings
from core.config.warnings import deprecated
from core.model import logger
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelMetadata import SimpleModelMetadata
from core.model.ModelPath import ModelPath
from core.model.models_constants import (
    B_PARAM_SIZE_PATTERN,
    M_PARAM_SIZE_PATTERN,
    validate_from,
)


class ModelConfigException(Exception):
    pass


class ModelParameters(BaseModel):
    num_ctx: int = Field(
        default=4096,
        description="Sets the size of the context window used to generate the next token. (Default: 4096)",
    )
    repeat_last_n: int = Field(
        default=64,
        description="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
    )
    repeat_penalty: float = Field(
        default=1.1,
        description="Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)",
    )
    temperature: float = Field(
        default=0.7,
        description="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)",
    )
    seed: int = Field(
        default=42,
        description="Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)",
    )
    stop: str = Field(
        default="AI assistant:",
        description="Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile.",
    )
    num_predict: int = Field(
        default=42,
        description="Maximum number of tokens to predict when generating text. (Default: -1, infinite generation)",
    )
    top_k: int = Field(
        default=40,
        description="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
    )
    top_p: float = Field(
        default=0.9,
        description="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
    )
    min_p: float = Field(
        default=0.05,
        description="Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0)",
    )


class FullModelParameters(ModelParameters):
    enable_thinking: bool
    max_new_tokens: int
    frequency_penalty: float
    presence_penalty: float
    mirostat: bool
    mirostat_tau: float
    mirostat_eta: float

    @classmethod
    def ollama_override(
        cls,
        full_model_parameters: "FullModelParameters",
        ollama_options: Optional[Any],
        enable_thinking: bool = False,
    ) -> "FullModelParameters":
        params = full_model_parameters.model_dump()
        if ollama_options is not None:
            options_dict = (
                ollama_options
                if isinstance(ollama_options, dict)
                else ollama_options.model_dump(exclude_unset=True)
            )
            for key, value in options_dict.items():
                if value is not None:
                    if key in params:
                        params[key] = value
                    if key == "num_predict":
                        params["max_new_tokens"] = value
        params["enable_thinking"] = enable_thinking
        return cls(**params)

    @classmethod
    def create(cls, model_file_info: ModelFileInfo):
        raise NotImplementedError

    @classmethod
    def load(cls, model_file_info: ModelFileInfo):
        raise NotImplementedError

    def save(self, model_file_info: ModelFileInfo):
        raise NotImplementedError


class MinimalModelConfig(BaseModel):
    HUGGINGFACE_PATH: Optional[str] = Field(
        default=None, description="Hugging Face repository path"
    )
    OLLAMA_PATH: Optional[str] = Field(
        default=None, description="Ollama repository path"
    )

    # @deprecated("use core.model.ModelFile.save instead.", category=DeprecationWarning, stacklevel=2)
    # def save(self, backend_model_file: str):
    #    with open(backend_model_file, "w") as f:
    #        for attr in self.__dict__:
    #            f.write(f"{attr.upper()}={self.__dict__[attr]}")


@deprecated(
    "use core.model.ModelFile.FullModelParameters instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
class MinimalTemperedModelConfig(MinimalModelConfig):
    temperature: Optional[float] = Field(
        default=get_settings().model.default_temperature,
        description="model temperature as float",
    )


@deprecated(
    "use core.model.ModelConfig.FullModelParameters instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
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

    @deprecated(
        "use core.model.ModelConfig.FullModelParameters.create instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    @classmethod
    def create(
        cls,
        model_path: ModelPath,
        model_metadata: SimpleModelMetadata,
        default_model_config: DefaultConfig,
    ):
        if model_metadata.model_type is not None:
            if model_path.model_type != model_metadata.model_type:
                raise ValueError(
                    f"Model type mismatch: {model_path.model_type} != {model_metadata.model_type}"
                )

        data: dict = model_path.__dict__.copy()

        if "endpoint_model_file" in data:
            b_param_pattern = re.search(
                B_PARAM_SIZE_PATTERN,
                data["endpoint_model_file"],
            )
            m_param_pattern = re.search(
                M_PARAM_SIZE_PATTERN,
                data["endpoint_model_file"],
            )

            if b_param_pattern or m_param_pattern:
                logger.warning(
                    f"Model {data['endpoint_model_file']} does not have any extension (is a model param size)."
                )
                data["FROM"] = validate_from(
                    ModelPath.compute_model_id(
                        model_name=data["model_name"],
                        endpoint_model_file=data["endpoint_model_file"],
                        is_ollama="ollama_path" in data,
                    )
                )
            else:
                data["FROM"] = validate_from(data["endpoint_model_file"])
                del data["endpoint_model_file"]
                from_ext = validate_from(data["FROM"]).split(".")[-1]
                if f".{from_ext}" != model_path.model_type.get_extension():
                    raise ValueError(
                        f"Model type mismatch: FROM extension {from_ext} != {model_path.model_type}"
                    )

        if "huggingface_path" in data:
            data["HUGGINGFACE_PATH"] = data["huggingface_path"]
            del data["huggingface_path"]

        data.update(model_metadata.__dict__)

        if "system_prompt" in data:
            data["SYSTEM"] = data["system_prompt"]
            del data["system_prompt"]

        default_data = {}
        for attr in default_model_config.__dict__:
            default_data[attr.removeprefix("default_")] = default_model_config.__dict__[
                attr
            ]
        data.update(default_data)

        return cls(**data)

    @deprecated(
        "use core.model.ModelConfig.FullModelParameters.load instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    @classmethod
    def load(
        cls, model_file_info: ModelFileInfo, modelfile_parameters: ModelParameters
    ):
        logger.debug(
            f"ModelConfig.load(model_file_info={model_file_info}, modelfile_parameters={modelfile_parameters})"
        )
        file_path: Path = model_file_info.modelfile_config_path
        logger.debug(f"ModelConfig.load(): file_path={file_path}")

        from_value = None
        filtered_content = ""
        with open(str(file_path), "r") as f:
            while line := f.readline():
                if line.startswith("#"):
                    if line.startswith("# FROM="):
                        endpoint_model_file = line.split("=", maxsplit=1)[1].strip()
                        if model_file_info.validate_FROM_with_endpoint_file(
                            FROM=endpoint_model_file
                        ):
                            raise ModelConfigException(
                                f"Model mismatch: {endpoint_model_file} != ({model_file_info.model_name}[:/]){model_file_info.endpoint_model_file}"
                            )
                        from_value = endpoint_model_file
                else:
                    filtered_content += line + "\n"
            json_data = json.loads(filtered_content)

        if from_value is None and "FROM" not in json_data:
            raise ModelConfigException(
                f"Model config file {file_path} does not contain FROM parameter."
            )

        json_data["FROM"] = validate_from(from_value or json_data["FROM"])

        json_data.update(
            modelfile_parameters.model_dump(
                exclude_unset=True, exclude_defaults=True, exclude_none=True
            )
        )
        json_data.update(model_file_info.model_dump())

        model: DefaultConfig = get_settings().model
        for default_key, value in model.model_dump().items():
            key = default_key.removeprefix("default_")
            if key is not None and key != "" and key not in json_data:
                json_data[key] = value

        logger.debug(f"ModelConfig.load(): json_data={json_data}")
        return ModelConfig(**json_data)

    def save(
        self, model_file_info: ModelFileInfo, modelfile_parameters: ModelParameters
    ):
        from core.model.ModelFile import MinimalModelFileContent

        logger.debug(
            f"ModelConfig.save(model_file_info={model_file_info}, modelfile_parameters={modelfile_parameters})"
        )
        file_path: Path = model_file_info.modelfile_config_path
        logger.debug(f"ModelConfig.save(): file_path={file_path}")
        exclude = {
            attr
            for attr, _ in list(ModelParameters.model_fields.items())
            + list(MinimalTemperedModelConfig.model_fields.items())
        }
        with open(str(file_path), "w") as f:
            model_dump = self.model_dump()
            if "FROM" not in model_dump:
                model_dump["FROM"] = model_file_info.model_id
            for attr, value in model_dump.items():
                if attr in MinimalModelFileContent.model_fields:
                    f.write(f"# {attr.upper()}={value}\n")
            f.write(
                self.model_dump_json(
                    indent=2,
                    exclude_unset=True,
                    exclude_defaults=True,
                    exclude_none=True,
                    exclude=exclude,
                )
            )
