from pydantic import BaseModel

from core.config import IncrementalConfigSchema


class PathsConfig(BaseModel):
    models: str = "models"
    logs: str = "logs"
    data: str = "data"
    src: str = "src"
    lib: str = "lib"
    temp: str = "temp"

    @staticmethod
    def add_schema(schema: IncrementalConfigSchema):
        paths = schema.add_section("paths", description="Path configuration")
        paths.path("models", "models", "Path to model files")
        paths.path("logs", "logs", "Path to log files")
        paths.path("data", "data", "Path to data files")
        paths.path("src", "src", "Path to source files")
        paths.path("lib", "lib", "Path to library files")
        paths.path("temp", "temp", "Path to temporary files")
