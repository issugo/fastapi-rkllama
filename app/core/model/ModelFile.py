import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.config import config_utils
from core.config.config_utils import get_settings
from core.config.warnings import deprecated
from core.model.Model import Model
from core.model.ModelConfig import (
    ModelConfig,
    ModelParameters,
    FullModelParameters,
    MinimalTemperedModelConfig,
    ModelConfigException,
)
from core.model.ModelFileInfo import ModelFileInfo
from core.model.models_constants import validate_from, DEFAULT_SYSTEM
from core.model.suppliers_model_info import OllamaModelInfo, HFModelInfo
from core.model.ModelMetadata import (
    SimpleModelMetadata,
    ModelMetadataFormat,
    ModelMetadata,
    ModelMetadataNotFoundException,
    BasicModelMetadata,
)
from core.model.ModelLicense import ModelLicense
from core.model.ModelPath import ModelPath, ModelNotFoundException, ModelException
from core.config.DefaultModelConfig import DefaultConfig

from core.model import logger
from core.model.OllamaManifest import OllamaManifest
from core.model.HfFileInfo import HfFileInfo

"""
Instruction 	Description
FROM (required) 	Defines the base model to use.
PARAMETER 	Sets the parameters for how Ollama will run the model.
TEMPLATE 	The full prompt template to be sent to the model.
SYSTEM 	Specifies the system message that will be set in the template.
ADAPTER 	Defines the (Q)LoRA adapters to apply to the model.
LICENSE 	Specifies the legal license.
MESSAGE 	Specify message history.

Parameter 	Description 	Value Type 	Example Usage
num_ctx 	Sets the size of the context window used to generate the next token. (Default: 4096) 	int 	num_ctx 4096
repeat_last_n 	Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx) 	int 	repeat_last_n 64
repeat_penalty 	Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1) 	float 	repeat_penalty 1.1
temperature 	The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8) 	float 	temperature 0.7
seed 	Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0) 	int 	seed 42
stop 	Sets the stop sequences to use. When this pattern is encountered the LLM will stop generating text and return. Multiple stop patterns may be set by specifying multiple separate stop parameters in a modelfile. 	string 	stop "AI assistant:"
num_predict 	Maximum number of tokens to predict when generating text. (Default: -1, infinite generation) 	int 	num_predict 42
top_k 	Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40) 	int 	top_k 40
top_p 	Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9) 	float 	top_p 0.9
min_p 	Alternative to the top_p, and aims to ensure a balance of quality and variety. The parameter p represents the minimum probability for a token to be considered, relative to the probability of the most likely token. For example, with p=0.05 and the most likely token having a probability of 0.9, logits with a value less than 0.045 are filtered out. (Default: 0.0) 	float 	min_p 0.05
"""


class ModelFileException(Exception):
    pass


class MinimalModelFileContent(BaseModel):
    FROM: str = Field(
        description="Defines the base model to use, can be path to the model file or docker image format (<name>:<tag>)."
    )

    _model: Model | None = None

    @property
    def model(self) -> Model:
        if self._model is None:
            self._model = Model.from_model_path(
                ModelPath.from_model_id(validate_from(self.FROM))
            )
        return self._model

    @property
    def model_id(self) -> str:
        return self.model.model_path.model_id

    @property
    def model_name(self) -> str:
        return self.model.model_name


class ModelFileContent(MinimalModelFileContent):
    Instruction: Optional[str] = Field(default=None, description="Description")
    SYSTEM: Optional[str] = Field(
        default=None,
        description="Specifies the system message that will be set in the template.",
    )
    TEMPLATE: Optional[str] = Field(
        default=None, description="The full prompt template to be sent to the model."
    )
    ADAPTER: Optional[str] = Field(
        default=None, description="Defines the (Q)LoRA adapters to apply to the model."
    )
    LICENSE: Optional[str] = Field(
        default=None, description="Specifies the legal license."
    )
    MESSAGE: Optional[str] = Field(default=None, description="Specify message history.")

    # PARAMETER: Sets the parameters for how Ollama will run the model.
    PARAMETERS: Optional[ModelParameters] = Field(
        default=None,
        description="Sets the parameters for how Ollama will run the model.",
    )


class ModelFile(ModelFileContent):
    model_file_info: ModelFileInfo

    endpoint_model_config: Optional[ModelConfig] = Field(
        default=None, description="Model configuration"
    )
    volatile_endpoint_model_config: Optional[bool] = Field(
        default=False, description="Whether the model config is volatile (saved or not)"
    )

    _request_options: dict = None
    _options: dict = None

    _licence: Optional[ModelLicense] = None

    @property
    def license(self):
        supplier_model_file_info = None
        if self.ollama_model_info_exists:
            supplier_model_file_info = self.model_file_info.ollama_model_info
        if self.huggingface_model_info_exists:
            supplier_model_file_info = self.model_file_info.huggingface_model_info
        if self.LICENSE:
            if self._licence:
                return self._licence
            self._licence = ModelLicense.from_modelfile_license(
                license_text=self.LICENSE,
                supplier_model_file_info=supplier_model_file_info,
            )
        if supplier_model_file_info:
            if supplier_model_file_info.license:
                self._licence = supplier_model_file_info.license
                return self._licence
        return None

    @property
    def description(self) -> str:
        """use instruction or model description"""
        model_description = None
        if self.instruction:
            model_description = self.instruction
        elif self.model.model_metadata.description:
            model_description = self.model.model_metadata.description
        if model_description:
            desc_lines = model_description.split("\n")
            desc_comment = "\n".join(
                [f"{line}" for line in desc_lines[:5]]
            )  # First 5 lines only
            return desc_comment
        return ""

    @property
    def simple_model_metadata(self) -> SimpleModelMetadata:
        try:
            if self.model is not None:
                if self.model.get_metadata_format is not None and (
                    self.model.get_metadata_format() == ModelMetadataFormat.SIMPLE
                    or self.model.get_metadata_format() == ModelMetadataFormat.COMPLETE
                ):
                    return self.model.model_metadata
        except ModelNotFoundException as e:
            logger.warning(str(e), exc_info=True)
        logger.warning(f"Model metadata is not available for model {self.model_name}")
        raise ModelMetadataNotFoundException(self.model_name)

    @property
    def model_metadata(
        self,
    ) -> None | BasicModelMetadata | SimpleModelMetadata | ModelMetadata:
        if self.model is not None:
            return self.model.model_metadata
        logger.warning(f"Model metadata is not available for model {self.model_name}")
        raise ModelMetadataNotFoundException(self.model_name)

    @property
    def full_model_parameters(self) -> FullModelParameters:
        param_values = {}
        default_param_values = get_settings().model.model_dump()
        for default_attr, value in default_param_values.items():
            attr_name = default_attr.removeprefix("default_")
            if attr_name is not None and attr_name != "":
                param_values[attr_name] = value
        for attr, value in self.model_parameters.items():
            if attr in param_values:
                param_values[attr] = value
        return FullModelParameters(**param_values)

    @property
    def model_parameters(self) -> dict:
        param_values = {}
        if self.PARAMETERS is not None:
            for attr, value in self.PARAMETERS.model_dump().items():
                if attr in param_values:
                    param_values[attr] = value
        if self.endpoint_model_config is not None:
            for attr, value in self.endpoint_model_config.model_dump().items():
                if attr in param_values:
                    param_values[attr] = value
        return param_values

    @classmethod
    def create(
        cls,
        model_file_info: ModelFileInfo,
        default_model_config: DefaultConfig,
        model: Model,
        model_license: ModelLicense,
    ):
        """
        sample arg values
        model_file_info = {"model_name": "Qwen3-1.7B-rk3588-1.2.1-unsloth-16k", "model_type": "RKLLM",
                                   "endpoint_model_file": "Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm",
                                   "endpoint_model_file_size": 2391955766, "license": null,
                                   "huggingface_path": "dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k",
                                   "ollama_path": null, "system_prompt": ""}
        """
        try:
            logger.debug(
                f"Creating ModelFile: {model_file_info.model_name}/{model_file_info.endpoint_model_file}"
            )

            if model_file_info.model_type is None:
                error_msg = "model_file_info.model_type is None"
                logger.error(f"Creating ModelFile: {error_msg}")
                raise ModelFileException(error_msg)

            model_metadata: ModelMetadata | SimpleModelMetadata | None = (
                model.model_metadata
            )
            logger.debug(f"Creating ModelFile: model_metadata={model_metadata}")
            if model_metadata is None:
                logger.warning("Creating ModelFile: model_metadata is None")

            config_data = {}
            model_file_info_dump = model_file_info.model_dump()
            for attr in model_file_info_dump:
                match attr:
                    case "endpoint_model_file":
                        config_data["FROM"] = validate_from(
                            f"{model_file_info.model_name}/{model_file_info_dump[attr]}"
                        )
                    case "huggingface_path":
                        config_data["HUGGINGFACE_PATH"] = model_file_info_dump[attr]
                    case "ollama_path":
                        config_data["OLLAMA_PATH"] = model_file_info_dump[attr]
                    case "system_prompt":
                        config_data["SYSTEM"] = model_file_info_dump[attr]
                    case _:
                        default_model_config_dump = default_model_config.model_dump()
                        if attr in default_model_config_dump:
                            config_data[attr] = default_model_config_dump[attr]
                        else:
                            config_data[attr] = model_file_info_dump[attr]

            endpoint_model_config = None
            if model_file_info.modelfile_match:
                # ModelFile exists and matches
                endpoint_model_config_dump = ModelConfig.load(
                    model_file_info=model_file_info,
                    modelfile_parameters=ModelParameters(),
                )
                mf_data = {"endpoint_model_config": endpoint_model_config_dump}
                mf_data.update(model_file_info.model_dump())
                logger.debug(
                    f"Using existing ModelFile: {mf_data['endpoint_model_file']}"
                )
                return ModelFile.load(model_path=model_file_info)

            if not model_file_info.system_prompt:
                model_file_info.system_prompt = DEFAULT_SYSTEM
                logger.debug(
                    f"Using default system prompt: {model_file_info.system_prompt}"
                )

            mf_data = {}
            if model_file_info.modelfile_exists:
                # ModelFile is existing but not matching
                mf_data.update({"volatile_endpoint_model_config": True})
                logger.debug(f"ModelFile is existing but not matching: set {mf_data}")

            if model_metadata is None:
                endpoint_model_config = ModelConfig(**config_data)
                mf_data.update({"endpoint_model_config": endpoint_model_config})
                try:
                    model_metadata = model_file_info.simple_model_metadata
                    mf_data.update({"model_metadata": model_metadata})
                except Exception as e:
                    logger.error(f"Error computing model metadata: {e}", exc_info=True)
            else:
                endpoint_model_config = ModelConfig.create(
                    model_path=model_file_info,
                    model_metadata=model_metadata,
                    default_model_config=default_model_config,
                )
                mf_data.update(
                    {
                        "endpoint_model_config": endpoint_model_config,
                        "model_metadata": model_metadata,
                    }
                )

            logger.debug(f"computed mf_data={mf_data}")
            """
            sample output
            mf_data = {'endpoint_model_config': ModelConfig(FROM='Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm',
                                                            HUGGINGFACE_PATH='dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                                                            SYSTEM='Tu es un assistant artificiel.', temperature=0.5,
                                                            enable_thinking=False, num_ctx=16384, max_new_tokens=16384,
                                                            top_k=7, top_p=0.5, repeat_penalty=1.1,
                                                            frequency_penalty=0.0, presence_penalty=0.0, mirostat=False,
                                                            mirostat_tau=3.0, mirostat_eta=0.001),
                       'model_metadata': SimpleModelMetadata(name='qwen3', architecture='qwen3', quantization='w8a8',
                                                             quantization_opt=0, quantization_hybrid_ratio=0.0,
                                                             parameters=1700000000, context_length=4096,
                                                             system_prompt='Tu es un assistant artificiel.',
                                                             temperature=0.7, model_type= < ModelType.RKLLM: 'RKLLM' >)}
            """
            mf_data.update(
                {
                    "FROM": validate_from(model_file_info.model_id),
                    "model_file_info": model_file_info_dump,
                }
            )
            if endpoint_model_config:
                config_data.update(endpoint_model_config.model_dump())
            for attr in config_data:
                if attr == attr.upper():
                    mf_data[attr] = config_data[attr]

            logger.debug(f"completed mf_data={mf_data}")

            """
            sample output
            mf_data = {'volatile_endpoint_model_config': True, 
                       'endpoint_model_config': ModelConfig(
                            HUGGINGFACE_PATH='dulimov/Qwen3-0.6B-rk3588-1.2.1-unsloth-16k', OLLAMA_PATH=None, temperature=0.5,
                            enable_thinking=False, num_ctx=16384, max_new_tokens=16384, top_k=7, top_p=0.5, repeat_penalty=1.1,
                            frequency_penalty=0.0, presence_penalty=0.0, mirostat=False, mirostat_tau=3.0, mirostat_eta=0.001),
                       'model_metadata': SimpleModelMetadata(name='qwen3', architecture='qwen3', quantization='w8a8',
                                                             quantization_opt=0, quantization_hybrid_ratio=0.0,
                                                             parameters=600000000, context_length=4096,
                                                             system_prompt='Tu es un assistant artificiel.',
                                                             temperature=0.7,
                                                             model_type= < ModelType.RKLLM: 'RKLLM' >), 
                       'model_file_info': {
                                           'model_name': 'Qwen3-0.6B-rk3588-1.2.1-unsloth-16k',
                                           'model_format': < ModelType.RKLLM: 'RKLLM' >, 
                                           'endpoint_model_file': 'Qwen3-0.6B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm', 
                                           'endpoint_model_file_size': 952996582, 
                                           'license': None, 
                                           'huggingface_path': 'dulimov/Qwen3-0.6B-rk3588-1.2.1-unsloth-16k', 
                                           'ollama_path': None, 
                                           'system_prompt': ''
                       }
            }

            mf_data = {'endpoint_model_config': ModelConfig(FROM='Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm',
                                                            HUGGINGFACE_PATH='dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                                                            SYSTEM='Tu es un assistant artificiel.', temperature=0.5,
                                                            enable_thinking=False, num_ctx=16384, max_new_tokens=16384,
                                                            top_k=7, top_p=0.5, repeat_penalty=1.1,
                                                            frequency_penalty=0.0, presence_penalty=0.0, mirostat=False,
                                                            mirostat_tau=3.0, mirostat_eta=0.001),
                       'model_metadata': SimpleModelMetadata(name='qwen3', architecture='qwen3', quantization='w8a8',
                                                             quantization_opt=0, quantization_hybrid_ratio=0.0,
                                                             parameters=1700000000, context_length=4096,
                                                             system_prompt='Tu es un assistant artificiel.',
                                                             temperature=0.7,
                                                             model_type= < ModelType.RKLLM: 'RKLLM' >), 'model_name': 'Qwen3-1.7B-rk3588-1.2.1-unsloth-16k', 'model_type': < ModelType.RKLLM: 'RKLLM' >, 'endpoint_model_file_size': 2391955766, 'license': None, 'ollama_path': None, 'FROM': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm', 'HUGGINGFACE_PATH': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k', 'name': 'qwen3', 'architecture': 'qwen3', 'quantization': 'w8a8', 'quantization_opt': 0, 'quantization_hybrid_ratio': 0.0, 'parameters': 1700000000, 'context_length': 4096, 'temperature': 0.5, 'SYSTEM': 'Tu es un assistant artificiel.', 'default': None, 'enable_thinking': False, 'num_ctx': 16384, 'max_new_tokens': 16384, 'top_k': 7, 'top_p': 0.5, 'repeat_penalty': 1.1, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'mirostat': False, 'mirostat_tau': 3, 'mirostat_eta': 0.001}
            """

            """
            have to found attributes:
                model_name: str
                model_type: Optional[ModelType] = None
                endpoint_model_file: str
                endpoint_model_file_size: int
                license: Optional[ModelLicense] = None
                huggingface_path: Optional[str] = Field(default=None, description="Hugging Face repository path")
                ollama_path: Optional[str] = Field(default=None, description="Ollama repository path")
                endpoint_model_config: ModelConfig
                volatile_endpoint_model_config: bool = False
                model_metadata: SimpleModelMetadata
                request_options: dict = None
                options: dict = None

            """
            model_file: ModelFile = ModelFile(**mf_data)
            model_file.model_file_info = model_file_info
            model_file._model = model
            model_file._license = model_license
            return model_file

        except Exception as e:
            logger.error(f"Error creating ModelFile: {e}", exc_info=True)
            raise e

    @staticmethod
    def get_property_modelfile(model_name: str, property: str):
        """Get a specific property from the Modelfile of a model."""
        modelfile_path: Path = ModelPath.__modelfile_path(model_name)
        if not modelfile_path.exists():
            error_msg = (
                f"Modelfile not found for model '{model_name}': {modelfile_path}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(OSError(error_msg))

        modelfile: ModelFile = ModelFile.load(
            model_path=ModelPath(model_name=model_name)
        )
        modelfile_dict = modelfile.model_dump()

        # Retrieve the value of the property
        return modelfile_dict.get(property, None)

    @classmethod
    def clean_modelfile(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.clean_modelfile(model_path={model_path})")

    @classmethod
    def clean_metadata(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.clean_metadata(model_path={model_path})")

    @classmethod
    def clean_config(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.clean_config(model_path={model_path})")

    @classmethod
    def clean(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.clean(model_path={model_path})")
        cls.clean_modelfile(model_path)
        cls.clean_config(model_path)
        Model.clean(model_path)

    @classmethod
    def _load_modelfile(cls, model_path: ModelPath) -> Any:
        logger.debug(f"ModelFile._load_modelfile(model_path={model_path})")
        model_file_info: ModelFileInfo = ModelFileInfo(**model_path.model_dump())
        modelfile_path: Path = model_file_info.modelfile_path
        logger.debug(f"ModelFile._load_modelfile(): modelfile_path={modelfile_path}")

        field_names = [
            "Instruction",
            "FROM",
            "TEMPLATE",
            "# HUGGINGFACE_PATH",
            "# OLLAMA_PATH",
            "SYSTEM",
            "ADAPTER",
            "LICENSE",
            "MESSAGE",
        ]
        field_pos = 0

        mf_data = {}
        params_data = {}
        with open(str(modelfile_path), "r") as f:
            while line := f.readline():
                while True:
                    field_name = field_names[field_pos]
                    if field_name == "Instruction" and line.startswith("# "):
                        instruction = line[2:] + "\n"
                        while True:
                            line = f.readline()
                            if line.startswith("# "):
                                instruction += line[2:] + "\n"
                            else:
                                break
                        if len(instruction) > 0:
                            mf_data["Instruction"] = instruction
                            field_pos += 1
                            break
                    elif line.startswith(field_name + " ") or line.startswith(
                        field_name + "="
                    ):
                        value = line[len(field_name) + 1 :].strip()
                        if value is not None and value != "":
                            if field_name.startswith("# "):
                                field_name = field_name[2:]
                            if field_name in ["TEMPLATE", "LICENSE"]:
                                if value.startswith('"""') and value.endswith('"""'):
                                    value = value[3:-3]
                            elif field_name in ["SYSTEM", "MESSAGE"]:
                                if value.startswith('"') and value.endswith('"'):
                                    value = value[1:-1]
                            mf_data[field_name] = value
                            field_pos += 1
                            break
                    elif line.startswith("PARAMETER "):
                        key, value = line[len("PARAMETER ") :].strip().split(" ", 1)
                        params_data[key] = value
                        break
                    elif line == "":
                        # empty line, continue
                        break
                    else:
                        field_pos += 1
                if field_pos >= len(field_names):
                    break

        logger.debug(f"ModelFile._load_modelfile(): mf_data={mf_data}")
        logger.debug(f"ModelFile._load_modelfile(): params_data={params_data}")

        if model_file_info.validate_FROM_with_endpoint_file(FROM=mf_data["FROM"]):
            raise ModelFileException(
                f"Model mismatch: {mf_data['FROM']} != ({model_file_info.model_name}[:/]){model_file_info.endpoint_model_file}"
            )

        mf_data.update(params_data)
        minimal_endpoint_model_config: MinimalTemperedModelConfig = (
            MinimalTemperedModelConfig(**mf_data)
        )
        endpoint_model_config_data = minimal_endpoint_model_config.model_dump()
        # do update twice to override default_model_config values with values from Modelfile
        endpoint_model_config_data.update(params_data)
        model: DefaultConfig = get_settings().model
        for default_key, value in model.model_dump().items():
            key = default_key.removeprefix("default_")
            if key is not None and key != "" and key not in endpoint_model_config_data:
                endpoint_model_config_data[key] = value

        mf_data.update({"model_file_info": model_file_info.model_dump()})
        modelfile = ModelFile(**mf_data)
        modelfile.PARAMETERS = ModelParameters(**params_data)
        modelfile.endpoint_model_config = ModelConfig(**endpoint_model_config_data)
        return modelfile

    def load_config(self) -> Any:
        logger.debug("ModelFile.load_config()")
        try:
            # config is a merge of Modelfile parameters and overriden default_model_config values
            # but save is only non Modelfile parameters values, so have to pass a Modelfile when saving
            model_config: ModelConfig = ModelConfig.load(
                model_file_info=self.model_file_info,
                modelfile_parameters=self.PARAMETERS,
            )
            return model_config
        except ModelConfigException as e:
            error_msg = f"Error loading model config: {str(e)}"
            logger.error(f"ModelFile.load_config(): {error_msg}", exc_info=True)
            raise ModelFileException(error_msg) from e

    @classmethod
    def load(cls, model_path: ModelPath, model: Model = None) -> Any:
        logger.debug(f"ModelFile.load(model_path={model_path})")
        try:
            if model is None:
                model: Model = Model.load(model_path=model_path)
            modelfile: ModelFile = cls._load_modelfile(model_path)
            modelfile._model = model
            modelfile.endpoint_model_config = modelfile.load_config()
            return modelfile
        except ModelException as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(f"ModelFile.load(): {error_msg}", exc_info=True)
            raise ModelFileException(str(e)) from e
        except ModelConfigException as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(f"ModelFile.load(): {error_msg}", exc_info=True)
            raise ModelFileException(error_msg) from e

    def save_modelfile(self):
        logger.debug("self.save_modelfile()")
        modelfile_path: Path = self.model_file_info.modelfile_path
        logger.debug(f"ModelFile.save(): modelfile_path={modelfile_path}")

        with open(str(modelfile_path), "w") as f:
            if self.endpoint_model_config is not None:
                endpoint_model_config_dump = self.endpoint_model_config.model_dump()
                # overwrite config by Modelfile content, keep the undef
                endpoint_model_config_dump.update(self.model_dump())
                for field_name in [
                    "Instruction",
                    "FROM",
                    "TEMPLATE",
                    "HUGGINGFACE_PATH",
                    "OLLAMA_PATH",
                    "SYSTEM",
                    "ADAPTER",
                    "LICENSE",
                    "MESSAGE",
                ]:
                    if field_name == "Instruction":
                        value = endpoint_model_config_dump.get(field_name)
                        if value is not None:
                            for str_value in value.split("\n"):
                                f.write(f"# {str_value}\n")
                            f.write("\n\n")  # empty lines to mark instruction end
                    elif field_name.endswith("_PATH"):
                        value = endpoint_model_config_dump.get(field_name)
                        if value is not None:
                            f.write(f"# {field_name}={value}\n")
                    elif field_name in ["TEMPLATE", "LICENSE"]:
                        value = endpoint_model_config_dump.get(field_name)
                        if value is not None and str(value) != "":
                            f.write(f'{field_name} """{value}"""\n')
                    elif field_name in ["SYSTEM", "MESSAGE"]:
                        value = endpoint_model_config_dump.get(field_name)
                        if value is not None and str(value) != "":
                            f.write(f'{field_name} "{value}"\n')
                    else:
                        value = endpoint_model_config_dump.get(field_name)
                        if value is not None and str(value) != "":
                            f.write(f"{field_name} {value}\n")

                if self.PARAMETERS is None:
                    if self.endpoint_model_config is not None:
                        parameters_values = {}
                        for key, value in endpoint_model_config_dump.items():
                            if value is not None:
                                try:
                                    default_value = getattr(
                                        get_settings().model, f"default_{key}"
                                    )
                                except AttributeError:
                                    default_value = None
                                if default_value is not None and value == default_value:
                                    continue
                                parameters_values[key] = value
                        if len(parameters_values) > 0:
                            self.PARAMETERS = ModelParameters(**parameters_values)

                if self.PARAMETERS is not None:
                    parameters_values = self.PARAMETERS.model_dump()
                    for key, value in parameters_values.items():
                        if value is not None:
                            try:
                                default_value = getattr(
                                    get_settings().model, f"default_{key}"
                                )
                            except AttributeError:
                                default_value = None
                            if default_value is not None and value == default_value:
                                continue
                            f.write(f"PARAMETER {key} {value}\n")

    def content(self, save: bool = True):
        logger.debug("self.content()")
        modelfile_path: Path = self.model_file_info.modelfile_path
        logger.debug(f"ModelFile.content(): modelfile_path={modelfile_path}")

        if not modelfile_path.exists() and save:
            self.save_modelfile()
        if modelfile_path.exists():
            with open(str(modelfile_path), "r") as f:
                return f.read()
        raise FileNotFoundError(f"Modelfile not found: {modelfile_path}")

    def save_config(self):
        logger.debug("self.save_config()")
        # config is a merge of Modelfile parameters and overriden default_model_config values
        # but save is only non Modelfile parameters values, so have to pass a Modelfile when saving
        self.endpoint_model_config.save(
            model_file_info=self.model_file_info, modelfile_parameters=self.PARAMETERS
        )

    def save(self):
        logger.debug(
            f"ModelFile.save(model_path={ModelPath(**self.model_file_info.model_dump())})"
        )
        if self._model:
            self._model.save()
        self.save_config()
        self.save_modelfile()
        # TODO: catch Exception then clean all

    @property
    def huggingface_file_info_exists(self) -> bool:
        return self.model_file_info.huggingface_file_info_exists

    @property
    def huggingface_file_info(self) -> HfFileInfo | None:
        return self.model_file_info.huggingface_file_info

    @huggingface_file_info.setter
    def huggingface_file_info(self, value: HfFileInfo):
        self.model_file_info.huggingface_file_info = value

    @property
    def huggingface_model_info_exists(self) -> bool:
        return self.model_file_info.huggingface_model_info_exists

    @property
    def huggingface_model_info(self) -> HFModelInfo | None:
        return self.model_file_info.huggingface_model_info

    @huggingface_model_info.setter
    def huggingface_model_info(self, value: HFModelInfo):
        self.model_file_info.huggingface_model_info = value

    @property
    def ollama_file_info_exists(self) -> bool:
        return self.model_file_info.ollama_file_info_exists

    @property
    def ollama_file_info(self) -> OllamaManifest | None:
        return self.model_file_info.ollama_file_info

    @ollama_file_info.setter
    def ollama_file_info(self, value: OllamaManifest):
        self.model_file_info.ollama_file_info = value

    @property
    def ollama_model_info_exists(self) -> bool:
        return self.model_file_info.ollama_model_info_exists

    @property
    def ollama_model_info(self) -> OllamaModelInfo | None:
        return self.model_file_info.ollama_model_info

    @ollama_model_info.setter
    def ollama_model_info(self, value: OllamaModelInfo):
        self.model_file_info.ollama_model_info = value

    def lock_model(self) -> int:
        return self.model_file_info.lock_model()

    def unlock_model(self, lock_id: int) -> None:
        self.model_file_info.unlock_model(lock_id)

    def is_locked(self) -> bool:
        return self.model_file_info.is_locked()

    @deprecated("Use full_model_parameters instead")
    @property
    def full_options(self) -> dict:
        """
        Get model options from Modelfile or return default options if not found.

        Returns:
            A dictionary of model options
        """
        if not self._options:
            # Define default options in case of error
            self._options = config_utils.rkllama_config.model.__dict__

            # Get the Modelfile of the model
            self.file = os.path.join(self._model_dir, "Modelfile")

            # First overrride default values with the ModelFile Parameters
            if os.path.isfile(self.file):
                # Try to read the Modelfile
                with open(self.file, "r") as file:
                    # Looping through each line in the Modelfile
                    # and extracting key-value pairs
                    for line in file:
                        line = line.strip()
                        if "=" in line:
                            key, value = line.split("=", 1)
                            self._options[key.strip().lower()] = str(value).strip()

            # Override with request options if provided
            if self._request_options and isinstance(self._request_options, dict):
                for option, value in self._request_options.items():
                    # Override modelfile options with request options
                    self._options[option.strip().lower()] = str(value).strip()

        # Return the options dictionary
        return self._options
