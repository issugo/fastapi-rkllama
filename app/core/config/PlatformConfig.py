from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from core.model.models_constants import RK_TAGS_LIST


class PlatformProcessor(str, Enum):
    rk3588 = "rk3588"
    rk3576 = "rk3576"

# validation process
for platform_processor in PlatformProcessor:
    assert platform_processor.value in RK_TAGS_LIST, f"Missing {platform_processor.value} in RK_TAGS_LIST"


class PlatformConfig(BaseModel):
    processor: Annotated[PlatformProcessor, Field(default=PlatformProcessor.rk3588, description="Target processor")]

