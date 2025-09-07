import os
import time
import logging

from dotenv import load_dotenv
from pydantic import BaseModel

from core.rkllm.GlobalState import GLOBAL_STATE
from core import config, model
from core.config.ModelConfig import get_model_default_options
from core.rkllm.rkllm import RKLLM

# Get logger for this module
logger = logging.getLogger("core.model.ModelFile")


class ModelFileIdentifier(BaseModel):
    model_name: str = None

class ModelFileInfo(ModelFileIdentifier):
    huggingface_path: str
    model_file: str
    system_prompt: str = ""

class ModelFile(ModelFileInfo):
    file: str
    model_dir: str
    request_options = None
    options = None

    @classmethod
    def create_model(cls, model_file_info : ModelFileInfo):
        """Class method to create a model file - replaces the create_modelfile function"""
        return cls.create_modelfile(model_file_info)

    @staticmethod
    def create_modelfile(model_file_info : ModelFileInfo):
        model_file : ModelFile = ModelFile(model_file_info)
        struct_modelfile = f"""
FROM="{model_file_info.model_file}"

HUGGINGFACE_PATH="{model_file_info.huggingface_path}"

SYSTEM="{model_file_info.system_prompt}"

TEMPERATURE={config.get("model", "default_temperature")}

ENABLE_THINKING={config.get("model", "default_enable_thinking")}

NUM_CTX={config.get("model", "default_num_ctx")}

MAX_NEW_TOKENS={config.get("model", "default_max_new_tokens")}

TOP_K={config.get("model", "default_top_k")}

TOP_P={config.get("model", "default_top_p")}

REPEAT_PENALTY={config.get("model", "default_repeat_penalty")}

FREQUENCY_PENALTY={config.get("model", "default_frequency_penalty")}

PRESENCE_PENALTY={config.get("model", "default_presence_penalty")}

MIROSTAT={config.get("model", "default_mirostat")}

MIROSTAT_TAU={config.get("model", "default_mirostat_tau")}

MIROSTAT_ETA={config.get("model", "default_mirostat_eta")}


"""

        # Use config for models path
        # path = os.path.join(config.get_path("models"), From.replace('.rkllm', ''))
        path = os.path.join(config.get_path("models"), model_file_info.model_name)
    
        # Create the directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
    
        # Create the Modelfile and write the content
        with open(os.path.join(path, "Modelfile"), "w") as f:
            f.write(struct_modelfile)


    def load_model(self):
        # Use config for models path
        self.model_dir = os.path.join(config.get_path("models"), self.model_name)
        self.file = os.path.join(self.model_dir, "Modelfile")

        if not os.path.exists(self.model_dir):
            return None, f"Model directory '{self.model_name}' not found."

        if not os.path.exists(os.path.join(self.model_dir, "Modelfile")) and (
            self.huggingface_path is None and self.From is None
        ):
            return None, f"Modelfile not found in '{self.model_name}' directory."
        elif self.huggingface_path is not None and self.From is not None:
            ModelFile.create_modelfile(self)
            time.sleep(0.1)

        model_file = self.model_file
        huggingface_path = self.huggingface_path
        try:
            # Load modelfile
            load_dotenv(self.file, override=True)

            model_file = os.getenv("FROM")
            huggingface_path = os.getenv("HUGGINGFACE_PATH")
        except RuntimeError as e:
            logger.error(f"Error loading modelfile: {e}", exc_info=True)

        # View config Vars
        logger.info(f"FROM: {model_file}\nHuggingFace Path: {huggingface_path}")

        if not model_file or not huggingface_path:
            return None, "FROM or HUGGINGFACE_PATH not defined in Modelfile."

        # Get model parameters if not provided
        if not self.options:
            self.options = self.get_model_full_options()

        try:
            # Change value of model_id with huggingface_path
            GLOBAL_STATE.loaded_model_hfpath = huggingface_path
            model.rkllm_model = RKLLM(
                os.path.join(self.model_dir, model_file), self.model_dir, options=self.options
            )
        except RuntimeError as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return None, str(e)

        # return model and error message
        return model.rkllm_model, None


    def get_model_full_options(self) -> dict:
        """
        Get model options from Modelfile or return default options if not found.

        Returns:
            A dictionary of model options
        """
        if not self.options:

            # Define default options in case of error
            self.options = get_model_default_options()

            # Get the Modelfile of the model
            self.file = os.path.join(config.get_path("models"), self.model_name, "Modelfile")

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
