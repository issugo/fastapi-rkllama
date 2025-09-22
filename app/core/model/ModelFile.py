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
        try:
            logger.debug(f"Creating ModelFile: {model_file_info.model_name}/{model_file_info.endpoint_model_file}")

            config_data = {}
            for attr in model_file_info.__dict__:
                match attr:
                    case "endpoint_model_file":
                        config_data["FROM"] = model_file_info.__dict__[attr]
                    case "huggingface_path":
                        config_data["HUGGINGFACE_PATH"] = model_file_info.__dict__[attr]
                    case "system_prompt":
                        config_data["SYSTEM"] = model_file_info.__dict__[attr]
                    case _:
                        if attr in default_model_config.__dict__:
                            config_data[attr] = default_model_config.__dict__[attr]
                        else:
                            config_data[attr] = model_file_info.__dict__[attr]

            if model_file_info.modelfile_match:
                # ModelFile exists and matches
                mf_data={
                    'endpoint_model_config': ModelConfig.load(model_file_info)
                }
                mf_data.update(model_file_info.__dict__)
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
                    model_metadata: SimpleModelMetadata = model_file_info.simple_model_metadata
                    if model_file_info.huggingface_model_info_exists:
                        model_metadata.update_using_huggingface_model_info(model_file_info.huggingface_model_info)
                    elif model_file_info._ollama_model_info:
                        model_metadata.update_using_ollama_model_info(model_file_info.ollama_model_info)
            else:
                # from conversion
                model_metadata: SimpleModelMetadata = SimpleModelMetadata.from_complete(
                    metadata=ModelMetadata.load(metadata_path)
                )
            if model_metadata is None:
                mf_data.update({
                    'endpoint_model_config': ModelConfig(**config_data)
                })
            else:
                mf_data.update({
                    'endpoint_model_config': ModelConfig.create(
                        model_path=model_file_info,
                        model_metadata=model_metadata,
                        default_model_config=default_model_config),
                    'model_metadata': model_metadata
                })
            mf_data.update(model_file_info.__dict__)
            return ModelFile(**mf_data)

        except Exception as e:
            logger.error(f"Error creating ModelFile: {e}", exc_info=True)
            raise e


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
