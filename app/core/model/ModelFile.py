import os


from core.config import config_utils
from core.model.ModelConfig import ModelConfig
from core.model.ModelMetadata import SimpleModelMetadata, METADATA_FILENAME, ModelMetadataFormat, ModelMetadata
from core.model.ModelPath import ModelPath
from core.config.DefaultModelConfig import DefaultModelConfig

from core.model import logger

DEFAULT_SYSTEM = "Tu es un assistant artificiel."


class ModelFileInfo(ModelPath):
    system_prompt: str = ""
    _simple_model_metadata: SimpleModelMetadata = None

    @property
    def simple_model_metadata(self) -> SimpleModelMetadata:
        if self._simple_model_metadata:
            return self._simple_model_metadata

        # ompute metadata from endpoint_model_file name using ModelPath.extract_model_details
        data = SimpleModelMetadata.compute(
            model_path=self,
            model_details=self.extract_model_details(),
            system_prompt=self.system_prompt)
        return SimpleModelMetadata(**data)

    @simple_model_metadata.setter
    def simple_model_metadata(self, value: SimpleModelMetadata):
        self._simple_model_metadata = value

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

class ModelFile(ModelFileInfo):
    endpoint_model_config: ModelConfig
    volatile_endpoint_model_config: bool = False
    model_metadata: SimpleModelMetadata
    request_options: dict = None
    options: dict = None

    @property
    def simple_model_metadata(self) -> SimpleModelMetadata:
        return self.model_metadata


    @classmethod
    def create(cls, model_file_info: ModelFileInfo, default_model_config: DefaultModelConfig):
        """
        sample arg values
        model_file_info = {"model_name": "Qwen3-1.7B-rk3588-1.2.1-unsloth-16k", "model_type": "RKLLM",
                                   "endpoint_model_file": "Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm",
                                   "endpoint_model_file_size": 2391955766, "license": null,
                                   "huggingface_path": "dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k",
                                   "ollama_path": null, "system_prompt": ""}
        """
        try:
            logger.debug(f"Creating ModelFile: {model_file_info.model_name}/{model_file_info.endpoint_model_file}")

            config_data = {}
            model_file_info_dump = model_file_info.model_dump()
            for attr in model_file_info_dump:
                match attr:
                    case "endpoint_model_file":
                        config_data["FROM"] = model_file_info_dump[attr]
                    case "huggingface_path":
                        config_data["HUGGINGFACE_PATH"] = model_file_info_dump[attr]
                    case "system_prompt":
                        config_data["SYSTEM"] = model_file_info_dump[attr]
                    case _:
                        default_model_config_dump = default_model_config.model_dump()
                        if attr in default_model_config_dump:
                            config_data[attr] = default_model_config_dump[attr]
                        else:
                            config_data[attr] = model_file_info_dump[attr]

            if model_file_info.modelfile_match:
                # ModelFile exists and matches
                mf_data={
                    'endpoint_model_config': ModelConfig.load(model_file_info)
                }
                mf_data.update(model_file_info.model_dump())
                logger.debug(f"Using existing ModelFile: {mf_data['endpoint_model_file']}")
                return ModelFile(**mf_data)

            if not model_file_info.system_prompt:
                model_file_info.system_prompt = DEFAULT_SYSTEM
                logger.debug(f"Using default system prompt: {model_file_info.system_prompt}")

            mf_data={}
            if model_file_info.modelfile_exists:
                # ModelFile is existing but not matching
                mf_data.update({
                    'volatile_endpoint_model_config': True
                })
                logger.debug(f"ModelFile is existing but not matching: set {mf_data}")

            metadata_path=os.path.join(model_file_info.model_dir, METADATA_FILENAME)
            md_format = ModelMetadataFormat.get_format(metadata_path)
            logger.debug(f"Searching for Metadata: get {md_format}")
            if md_format == ModelMetadataFormat.SIMPLE:
                model_metadata: SimpleModelMetadata = SimpleModelMetadata.load(metadata_path)
            elif md_format is None:
                if model_file_info.huggingface_model_info_exists and model_file_info.ollama_model_info_exists:
                    model_metadata: SimpleModelMetadata = SimpleModelMetadata.from_complete(
                        metadata=ModelMetadata.build(
                            hf_model_info=model_file_info.huggingface_model_info,
                            ollama_model_info=model_file_info.ollama_model_info
                        )
                    )
                else:
                    model_metadata: SimpleModelMetadata | None = None
                    if model_file_info.huggingface_model_info_exists:
                        model_metadata = SimpleModelMetadata.create_using_huggingface_model_info(
                            model_metadata_data=SimpleModelMetadata.compute(
                                model_path=model_file_info,
                                model_details=model_file_info.extract_model_details(),
                                system_prompt=model_file_info.system_prompt
                            ),
                            huggingface_model_info=model_file_info.huggingface_model_info)
                    elif model_file_info.ollama_model_info_exists:
                        model_metadata = SimpleModelMetadata.create_using_ollama_model_info(
                            model_metadata_data=SimpleModelMetadata.compute(
                                model_path=model_file_info,
                                model_details=model_file_info.extract_model_details(),
                                system_prompt=model_file_info.system_prompt
                            ),
                            ollama_model_info=model_file_info.ollama_model_info)
            else:
                # from conversion
                model_metadata: SimpleModelMetadata = SimpleModelMetadata.from_complete(
                    metadata=ModelMetadata.load(metadata_path)
                )
            if model_metadata is None:
                mf_data.update({
                    'endpoint_model_config': ModelConfig(**config_data)
                })
                try:
                    model_metadata = model_file_info.simple_model_metadata
                    mf_data.update({
                        'model_metadata': model_metadata
                    })
                except Exception as e:
                    logger.error(f"Error computing model metadata: {e}", exc_info=True)
            else:
                mf_data.update({
                    'endpoint_model_config': ModelConfig.create(
                        model_path=model_file_info,
                        model_metadata=model_metadata,
                        default_model_config=default_model_config),
                    'model_metadata': model_metadata
                })

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
            mf_data.update(model_file_info_dump)
            logger.debug(f"completed mf_data={mf_data}")
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
            return ModelFile(**mf_data)

        except Exception as e:
            logger.error(f"Error creating ModelFile: {e}", exc_info=True)
            raise e

    @classmethod
    def clean(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.clean(model_path={model_path})")
        # TODO: rm Modelfile
        # TODO: rm Metadata
        # TODO: rm config

    @classmethod
    def load(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.load(model_path={model_path})")
        # TODO: load Modelfile
        # TODO: load Metadata
        # TODO: load config
        # TODO: create the ModelFile object
        # TODO: catch Exception then log and rethrow

    def save(self):
        logger.debug(f"ModelFile.save(model_path={ModelPath(**self.model_dump())})")
        # TODO: save config
        # TODO: save Metadata
        # TODO: save Modelfile
        # TODO: catch Exception then clean all

    @property
    def full_options(self) -> dict:
        """
        Get model options from Modelfile or return default options if not found.

        Returns:
            A dictionary of model options
        """
        if not self.options:

            # Define default options in case of error
            self.options = config_utils.rkllama_config.model.__dict__

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
                            self.options[key.strip().lower()] = str(value).strip()

            # Override with request options if provided
            if self.request_options and isinstance(self.request_options, dict):
                for option, value in self.request_options.items():
                    # Override modelfile options with request options
                    self.options[option.strip().lower()] = str(value).strip()

        # Return the options dictionary
        return self.options

    # TODO: create a dump method

def get_property_modelfile(model_name: str, property: str, models_path: str = "models"):
    """Get a specific property from the Modelfile of a model."""
    modelfile = os.path.join(models_path, model_name, "Modelfile")

    # Initialize an empty dictionary to store key-value pairs
    modelfile_dict = {}

    # Open and read the file
    try:
        with open(modelfile, "r") as file:
            for line in file:
                line = line.strip()
                if "=" in line:
                    # Split the line into key and value (split on first '=')
                    key, value = line.split("=", 1)
                    modelfile_dict[key] = value
    except FileNotFoundError:
        logger.error(f"Error: File '{modelfile}' not found.")

    # Retrieve the value of the property
    return modelfile_dict.get(property, None)
