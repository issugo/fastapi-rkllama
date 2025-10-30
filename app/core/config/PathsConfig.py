from enum import Enum
from typing import Annotated
from core.config.warnings import deprecated

from pydantic import BaseModel, Field

class PATH_KEY(str, Enum):
    MODELS = "models"
    LOGS = "logs"
    DATA = "data"
    SRC = "src"
    LIB = "lib"
    TEMP = "temp"

class Paths(BaseModel):
    models: Annotated[str, Field(default="models", description="Path to model files")]
    logs: Annotated[str, Field(default="logs", description="Path to log files")]
    data: Annotated[str, Field(default="data", description="Path to data files")]
    src: Annotated[str, Field(default="src", description="Path to source files")]
    lib: Annotated[str, Field(default="lib", description="Path to library files")]
    temp: Annotated[str, Field(default="temp", description="Path to temporary files")]

@deprecated("use core.config.Paths instead.", category=DeprecationWarning, stacklevel=2)
class PathsConfig(Paths):
    pass

