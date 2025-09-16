from typing import Optional

from pydantic import BaseModel

from core.model.ModelType import ModelType


class ModelDetails(BaseModel):
    format: Optional[str]
    family: Optional[str]
    parameter_size: str   # ex: 3B
    quantization_level: str

class ModelInfo(BaseModel):
    name: str  # Use simplified name like qwen:3b
    model: str # Match Ollama's format
    modified_at: str
    size: int
    digest: str = "" # Ollama field (not used but included for compatibility)
    details: ModelDetails
    model_type: ModelType