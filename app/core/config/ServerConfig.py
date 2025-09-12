from typing import Annotated

from pydantic import BaseModel, Field

class ServerConfig(BaseModel):
    port: Annotated[int, Field( default=8080, description="Server port number", gt=0, le=65535)]
    host: Annotated[str, Field( default="0.0.0.0", description="Server host address")]
    debug: Annotated[bool, Field( default=False, description="Enable debug mode")]
