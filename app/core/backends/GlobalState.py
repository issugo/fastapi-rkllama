from typing import Union, Optional

from pydantic import BaseModel

from core.model.ModelPath import ModelPath
from core.model.Model import Model


class GlobalState(BaseModel):
    rkllm_model: Optional[Model] = None  # Model instance

    @property
    def current_model(self) -> Union[ModelPath | None]:
        """Global variable for storing the loaded model"""
        if self.rkllm_model:
            return self.rkllm_model.model_file
        return None

    @property
    def loaded_model_hfpath(self) -> Union[str | None]:
        """Global variable for storing the loaded model"""
        if self.rkllm_model:
            return self.rkllm_model.model_file.huggingface_path
        return None


GLOBAL_STATE = GlobalState()


def unload_model():
    if GLOBAL_STATE.rkllm_model:
        GLOBAL_STATE.rkllm_model.release()
        GLOBAL_STATE.rkllm_model = None
