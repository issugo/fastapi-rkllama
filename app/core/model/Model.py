import os
import re
import threading
from typing import Union

from pydantic import BaseModel

from core.model.ModelFile import ModelFile
from core.backends.rkllm.rkllm_endpoint import RKLLMEndpoint
from core.model.ModelInfo import ModelDetails
from core.model.ModelPath import ModelType
from core.model.converter.quantization_constants import quant_mapping, quant_patterns


class ModelSharedData(BaseModel):
    global_status = -1
    global_text = []


class Model(BaseModel):
    model_file: Union[ModelFile | None] = None
    endpoint: Union[RKLLMEndpoint | None] = None
    shared_data: ModelSharedData = ModelSharedData()
    usage_lock: threading.Lock = threading.Lock()  # old verrou

    def unload(self):
        if self.endpoint:
            self.endpoint.release()
            self.endpoint = None
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
                details["quantization_level"] = quant_mapping.get(quant_type, quant_type)
                break

        return details
