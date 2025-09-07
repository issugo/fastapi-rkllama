from pydantic import BaseModel


class GlobalState(BaseModel):
    global_status = -1
    global_text = []
    loaded_model_hfpath = ""

GLOBAL_STATE = GlobalState()
