from pydantic import BaseModel


class ModelPath(BaseModel):
    path: str
