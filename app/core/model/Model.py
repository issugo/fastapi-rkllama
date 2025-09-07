import threading
from typing import Union

from pydantic import BaseModel

from core.model.ModelFile import ModelFile
from core.rkllm.rkllm import RKLLM

class ModelSharedData(BaseModel):
    global_status = -1
    global_text = []


class Model(BaseModel):
    model_file: Union[ModelFile|None] = None
    rkllm_model: Union[RKLLM|None] = None
    shared_data: ModelSharedData = ModelSharedData()
    usage_lock: threading.Lock = threading.Lock() # old verrou

    def unload(self):
        if self.rkllm_model:
            self.rkllm_model.release()
            self.rkllm_model = None
        self.model_file = None


