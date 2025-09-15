from typing import Optional

from pydantic import BaseModel

from core.model.ModelType import ModelType


class ModelDetails(BaseModel):
    format: Optional[str]
    family: Optional[str]
    parameter_size: str
    quantization_level: str

class ModelInfo(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str = ""
    details: ModelDetails
    model_type: ModelType