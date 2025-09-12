from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

class PlatformProcessor(str, Enum):
    rk3588 = "rk3588"
    rk3576 = "rk3576"

class PlatformConfig(BaseModel):
    processor: Annotated[PlatformProcessor, Field(default=PlatformProcessor.rk3588, description="Target processor")]

