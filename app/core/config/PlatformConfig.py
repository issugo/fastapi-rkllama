from pydantic import BaseModel

from core.config import IncrementalConfigSchema


class PlatformConfig(BaseModel):
    processor: str = "rk3588"

    @staticmethod
    def add_schema(schema: IncrementalConfigSchema):
        platform = schema.add_section("platform", description="Platform configuration")
        platform.string(
            "processor", "rk3588", "Target processor", options=["rk3588", "rk3576"]
        )
