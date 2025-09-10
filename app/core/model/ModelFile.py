import json
import os

from pydantic.v1.json import pydantic_encoder

import core.config.config_utils
from core.model.ModelPath import ModelPath
from core import config
from core.config.ModelConfig import get_model_default_options

from core.model import logger

DEFAULT_SYSTEM = "Tu es un assistant artificiel."


class ModelFileInfo(ModelPath):
    system_prompt: str = ""

class ModelFile(ModelFileInfo):
    request_options: dict = None
    options: dict = None

    @property
    def file(self):
        return os.path.join(self.model_dir, "Modelfile")

    @classmethod
    def create_model(cls, model_file_info : ModelFileInfo):
        model_file : ModelFile = ModelFile(**json.loads(json.dumps(model_file_info, default=pydantic_encoder)))
        struct_modelfile = f"""
FROM="{model_file_info.endpoint_model_file}"

HUGGINGFACE_PATH="{model_file_info.huggingface_path}"

SYSTEM="{model_file_info.system_prompt}"

TEMPERATURE={core.config.config_utils.get("model", "default_temperature")}

ENABLE_THINKING={core.config.config_utils.get("model", "default_enable_thinking")}

NUM_CTX={core.config.config_utils.get("model", "default_num_ctx")}

MAX_NEW_TOKENS={core.config.config_utils.get("model", "default_max_new_tokens")}

TOP_K={core.config.config_utils.get("model", "default_top_k")}

TOP_P={core.config.config_utils.get("model", "default_top_p")}

REPEAT_PENALTY={core.config.config_utils.get("model", "default_repeat_penalty")}

FREQUENCY_PENALTY={core.config.config_utils.get("model", "default_frequency_penalty")}

PRESENCE_PENALTY={core.config.config_utils.get("model", "default_presence_penalty")}

MIROSTAT={core.config.config_utils.get("model", "default_mirostat")}

MIROSTAT_TAU={core.config.config_utils.get("model", "default_mirostat_tau")}

MIROSTAT_ETA={core.config.config_utils.get("model", "default_mirostat_eta")}


"""

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
            self.options = get_model_default_options()

            # Get the Modelfile of the model
            self.file = os.path.join(core.config.config_utils.get_path("models"), self.model_name, "Modelfile")

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
