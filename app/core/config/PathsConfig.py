from typing import Annotated

from pydantic import BaseModel, Field

class PathsConfig(BaseModel):
    models: Annotated[str, Field(default="models", description="Path to model files")]
    logs: Annotated[str, Field(default="logs", description="Path to log files")]
    data: Annotated[str, Field(default="data", description="Path to data files")]
    src: Annotated[str, Field(default="src", description="Path to source files")]
    lib: Annotated[str, Field(default="lib", description="Path to library files")]
    temp: Annotated[str, Field(default="temp", description="Path to temporary files")]

