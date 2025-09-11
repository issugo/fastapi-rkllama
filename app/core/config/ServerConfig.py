from pydantic import BaseModel

from core.config import IncrementalConfigSchema


class ServerConfig(BaseModel):
    port: int = 8080
    host: str = "0.0.0.0"
    debug: bool = False

    @staticmethod
    def add_schema(schema: IncrementalConfigSchema):
        server = schema.add_section("server", description="Server configuration settings")
        server.integer("port", 8080, "Server port number", min_value=1, max_value=65535)
        server.string("host", "0.0.0.0", "Server host address")
        server.boolean("debug", False, "Enable debug mode")
