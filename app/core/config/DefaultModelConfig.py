from typing import Any, Optional, Annotated
from core.config.warnings import deprecated

from pydantic import BaseModel, Field

class DefaultConfig(BaseModel):
    default: Annotated[Optional[str], Field(default=None, description="Default model to use")] = None
    default_temperature: Annotated[float, Field(default=0.5, description="Default temperature for the model to use")]
    default_enable_thinking: Annotated[bool, Field(default=False, description="Default Enable Thinking for the model to use")]
    default_num_ctx: Annotated[int, Field(default=16384, description="Default Context Length for the model to use")]
    default_max_new_tokens: Annotated[int, Field(default=16384, description="Default Max New Tokens for the model to use")]
    default_top_k: Annotated[int, Field(default=7, description="Default Top K for the model to use")]
    default_top_p: Annotated[float, Field(default=0.5, description="Default Top P for the model to use")]
    default_repeat_penalty: Annotated[float, Field(default=1.1, description="Default Repeat Penalty for the model to use")]
    default_frequency_penalty: Annotated[float, Field(default=0.0, description="Default Frequency Penalty for the model to use")]
    default_presence_penalty: Annotated[float, Field(default=0.0, description="Default Presence Penalty for the model to use")]
    default_mirostat: Annotated[bool, Field(default=False, description="Default Mirostat for the model to use")]
    default_mirostat_tau: Annotated[float, Field(default=3, description="Default Mirostat Tau for the model to use")]
    default_mirostat_eta: Annotated[float, Field(default=0.001, description="Default Mirostat Eta for the model to use")]

@deprecated("use core.config.DefaultConfig instead.", category=DeprecationWarning, stacklevel=2)
class DefaultModelConfig(DefaultConfig):
    pass

