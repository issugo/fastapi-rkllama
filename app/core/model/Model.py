from pydantic import BaseModel

from core import model
from core.model import ModelName


class Model(BaseModel):
    name: ModelName


# TODO: move as method "unload" in Model
def unload_model():
    if model.modele_rkllm:
        model.modele_rkllm.release()
        model.modele_rkllm = None
