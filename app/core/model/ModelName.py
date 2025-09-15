import os
from typing import Optional

from pydantic import BaseModel

import core.config.config_utils
from core.model.ModelType import ModelType


class ModelName(BaseModel):
    model_name: str
    model_type: Optional[ModelType] = None


def get_model_size(model_name) -> int:
    """
    Get the size of a model
    Args:
        model_name: The name of the model directory
    Returns:
        The size of the model in bytes or None if not found
    """

    # Get the models directory
    models_dir = core.config.config_utils.get_path("models")
    model_path = os.path.join(models_dir, model_name)

    # check for the RKLLM file to get his size
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith(".rkllm"):
                size = os.path.getsize(os.path.join(model_path, file))
                return size

    return None

