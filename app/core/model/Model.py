import os
import time
import re
import threading
from typing import Union, Tuple, Any, List, Optional

from pydantic import BaseModel
from dotenv import load_dotenv

import core.config.config_utils
from core.model.ModelFile import ModelFile
from core.backends.backend import Backend
from core.model.ModelInfo import ModelDetails
from core.model.ModelPath import ModelType
from core.model.converter.quantization_constants import quant_mapping, quant_patterns

from core.model import logger

class ModelSharedData(BaseModel):
    global_status: int = -1
    global_text: List[str] = []

class ModelException(Exception):
    pass

class Model(BaseModel):
    model_file: ModelFile
    #backend: Optional[Backend]
    #shared_data: ModelSharedData
    #usage_lock: threading.Lock

    def __init__(self, model_file: ModelFile, /, **data: Any):
        super().__init__(**data)

        if not os.path.exists(model_file.model_dir):
            err_msg = f"Model directory '{model_file.model_name}' not found."
            logger.error(f"Error loading model: {err_msg}", exc_info=True)
            raise ModelException(err_msg)

        if not os.path.exists(os.path.join(model_file.model_dir, "Modelfile")) and (
            model_file.huggingface_path is None and model_file.From is None
        ):
            err_msg = f"Modelfile not found in '{model_file.model_name}' directory."
            logger.error(f"Error loading model: {err_msg}", exc_info=True)
            raise ModelException(err_msg)
        elif model_file.huggingface_path is not None and model_file.From is not None:
            ModelFile.create_modelfile(model_file)
            time.sleep(0.1)

        endpoint_model_file = model_file.endpoint_model_file
        huggingface_path = model_file.huggingface_path
        try:
            # Load modelfile
            load_dotenv(model_file.file, override=True)

            endpoint_model_file = os.getenv("FROM", endpoint_model_file)
            huggingface_path = os.getenv("HUGGINGFACE_PATH", huggingface_path)
        except RuntimeError as e:
            logger.error(f"Error loading modelfile: {e}", exc_info=True)
            raise ModelException(e)

        # View config Vars
        logger.info(f"FROM: {endpoint_model_file}\nHuggingFace Path: {huggingface_path}")

        if not endpoint_model_file or not huggingface_path:
            err_msg = "FROM or HUGGINGFACE_PATH not defined in Modelfile."
            logger.error(f"Error loading model: {err_msg}", exc_info=True)
            raise ModelException(err_msg)

        # Get model parameters if not provided
        if not model_file.options:
            model_file.options = model_file.full_options

        try:
            from core.backends.GlobalState import GLOBAL_STATE
            from core.backends.rkllm.rkllm_backend import RKLLMBackend

            # Change value of model_id with huggingface_path
            GLOBAL_STATE.loaded_model_hfpath = huggingface_path
            GLOBAL_STATE.backend = RKLLMBackend(
                model_path=os.path.join(model_file.model_dir, endpoint_model_file),
                model_dir=self.model_dir,
                options=self.options
            )
        except RuntimeError as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise ModelException(e)

        self.model_file = model_file

        self.shared_data = ModelSharedData()
        self.usage_lock = threading.Lock()  # old verrou

    def unload(self):
        if self.backend:
            self.backend.release()
            self.backend = None
        self.model_file = None

    @staticmethod
    def extract_model_details(model_type: ModelType, model_name) -> ModelDetails:
        """
        Extract model parameter size and quantization type from model name

        Args:
            model_name: Model name or file path

        Returns:
            Dictionary with parameter_size and quantization_level
        """
        # Initialize default values
        details = ModelDetails(parameter_size="Unknown", quantization_level="Unknown")

        # Remove path and extension if present
        if isinstance(model_name, str):
            basename = os.path.basename(model_name).replace(model_type.get_extension(), "")
        else:
            basename = str(model_name)

        # Extract parameter size (e.g., 3B, 7B, 13B)
        param_size_match = re.search(r"(\d+\.?\d*)(b|B)", basename)
        if param_size_match:
            size = param_size_match.group(1)
            # Convert to standard format (3B, 7B, 13B, etc)
            if "." in size:
                # For sizes like 1.1B, 2.7B
                details["parameter_size"] = f"{size}B"
            else:
                # For sizes like 3B, 7B
                details["parameter_size"] = f"{size}B"

        for quant_type, pattern in quant_patterns:
            if re.search(pattern, basename, re.IGNORECASE):
                # Use Ollama-style quantization name if available
                details["quantization_level"] = core.config.config_utils.get(quant_type, quant_type)
                break

        return details
