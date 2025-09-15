import json
import os

from pydantic.v1.json import pydantic_encoder

from core.config import config_utils
from core.config.config_utils import rkllama_config
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

        # TODO: compute metadata from endpoint_model_file name using ModelPath.extract_model_details
        data={}
        return SimpleModelMetadata(**data)

class ModelFile(ModelFileInfo):
    endpoint_model_config: ModelConfig
    model_metadata: SimpleModelMetadata
    request_options: dict = None
    options: dict = None

    @property
    def simple_model_metadata(self) -> SimpleModelMetadata:
        return self.model_metadata


    @classmethod
    def create(cls, model_file_info : ModelFileInfo, model_config: DefaultModelConfig):
        data = {}
        for attr in model_file_info.__dict__:
            match attr:
                case "endpoint_model_file":
                    data["FROM"] = model_file_info.__dict__[attr]
                case "huggingface_path":
                    data["HUGGINGFACE_PATH"] = model_file_info.__dict__[attr]
                case "system_prompt":
                    data["SYSTEM"] = model_file_info.__dict__[attr]
                case _:
                    data[attr] = model_config.__dict__[attr]

        model_file_info = ModelFileInfo(**data)
        if model_file_info.modelfile_match:
            mf_data={
                'endpoint_model_config': ModelConfig.load(model_file_info)
            }
            mf_data.update(model_file_info.__dict__)
            return ModelFile(**mf_data)

        if not model_file_info.system_prompt:
            model_file_info.system_prompt = DEFAULT_SYSTEM

        if not model_file_info.modelfile_exists:
            metadata_path=os.path.join(model_file_info.model_dir, METADATA_FILENAME)
            md_format = ModelMetadataFormat.get_format(metadata_path)
            if md_format == ModelMetadataFormat.SIMPLE:
                model_metadata: SimpleModelMetadata = SimpleModelMetadata.load(metadata_path)
            elif md_format is None:
                model_metadata: SimpleModelMetadata = model_file_info.simple_model_metadata
            else:
                # from conversion
                model_metadata: SimpleModelMetadata = SimpleModelMetadata.from_complete(
                    metadata=ModelMetadata.load(metadata_path)
                )
            mf_data={
                'endpoint_model_config': ModelConfig.create(
                    model_path=model_file_info,
                    model_metadata=model_metadata,
                    default_model_config=rkllama_config.model),
                'model_metadata': model_metadata
            }
            mf_data.update(model_file_info.__dict__)
            return ModelFile(**mf_data)

        struct_modelfile = model_config.struct_modelfile.format(**model_file_info.full_options)
        model_file : ModelFile = ModelFile(**json.loads(json.dumps(model_file_info, default=pydantic_encoder)))

        # Create the directory if it doesn't exist
        if not os.path.exists(model_file.model_dir):
            os.makedirs(model_file.model_dir)
    
        # Create the Modelfile and write the content
        with open(model_file.file, "w") as f:
            f.write(struct_modelfile)

        return model_file

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
                            self.options[key.lower().strip()] = str(value).strip()

            # Override with request options if provided
            if self.request_options and isinstance(self.request_options, dict):
                for option, value in self.request_options.items():
                    # Override modelfile options with request options
                    self.options[option.lower().strip()] = str(value).strip()

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
