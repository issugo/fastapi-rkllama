from typing import Optional

from pydantic import BaseModel

from core.model.ModelName import ModelType


class RKPullRequest(BaseModel):
    model: str
    model_name: Optional[str]
    model_type: Optional[ModelType] = None