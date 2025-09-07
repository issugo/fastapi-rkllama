import os
from typing import Union

from pydantic import BaseModel

from core.config import config
from core.model.ModelName import ModelName


class ModelPath(ModelName):
    huggingface_path: str
    rkllm_model_file: str
    _model_dir: Union[str|None] = None

    @property()
    def model_dir(self):
        if not self._model_dir:
            self._model_dir = os.path.join(config.get_path("models"), self.model_name.replace('.rkllm', ''))
        return self._model_dir