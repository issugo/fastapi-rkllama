from typing import Annotated
from core.config.warnings import deprecated

from pydantic import BaseModel, Field


class Server(BaseModel):
    port: Annotated[
        int, Field(default=8080, description="Server port number", gt=0, le=65535)
    ]
    host: Annotated[str, Field(default="0.0.0.0", description="Server host address")]
    debug: Annotated[bool, Field(default=False, description="Enable debug mode")]
    max_number_models_loaded_in_memory: Annotated[
        int, Field(default=1, description="Maximum number of models loaded into memory")
    ]
    max_minutes_loaded_in_memory: Annotated[
        int,
        Field(default=180, description="Maximum number of minutes loaded into memory"),
    ]


@deprecated(
    "use core.config.Server instead.", category=DeprecationWarning, stacklevel=2
)
class ServerConfig(Server):
    pass
