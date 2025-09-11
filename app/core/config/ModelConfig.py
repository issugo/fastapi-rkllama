from typing import Any, Optional

from pydantic import BaseModel

import core.config.config_utils
from core.config import IncrementalConfigSchema


class ModelConfig(BaseModel):
    default: Optional[str] = None
    default_temperature: float = 0.5
    default_enable_thinking: bool = False
    default_num_ctx: int = 16384
    default_max_new_tokens: int = 16384
    default_top_k: int = 7
    default_top_p: float = 0.5
    default_repeat_penalty: float = 1.1
    default_frequency_penalty: float = 0.0
    default_presence_penalty: float = 0.0
    default_mirostat: bool = False
    default_mirostat_tau: float = 3
    default_mirostat_eta: float = 0.001

    @staticmethod
    def add_schema(schema: IncrementalConfigSchema):
        model = schema.add_section("model", description="Model configuration")
        model.string("default", "", "Default model to use")
        model.float("default_temperature", 0.5, "Default temperature for the model to use")
        model.boolean(
            "default_enable_thinking", False, "Default Enable Thinking for the model to use"
        )
        model.integer(
            "default_num_ctx", 16384, "Default Context Length for the model to use"
        )
        model.integer(
            "default_max_new_tokens", 16384, "Default Max New Tokens for the model to use"
        )
        model.integer("default_top_k", 7, "Default Top K for the model to use")
        model.float("default_top_p", 0.5, "Default Top P for the model to use")
        model.float(
            "default_repeat_penalty", 1.1, "Default Repeat Penalty for the model to use"
        )
        model.float(
            "default_frequency_penalty",
            0.0,
            "Default Frequency Penalty for the model to use",
        )
        model.float(
            "default_presence_penalty", 0.0, "Default Presence Penalty for the model to use"
        )
        model.boolean("default_mirostat", False, "Default Mirostat for the model to use")
        model.float("default_mirostat_tau", 3, "Default Mirostat Tau for the model to use")
        model.float(
            "default_mirostat_eta", 0.1, "Default Mirostat Eta for the model to use"
        )




def get_model_default_options() -> dict[str | Any, Any]:
    default_options = {
        "temperature": core.config.config_utils.get("model", "default_temperature"),
        "num_ctx": core.config.config_utils.get("model", "default_num_ctx"),
        "max_new_tokens": core.config.config_utils.get("model", "default_max_new_tokens"),
        "top_k": core.config.config_utils.get("model", "default_top_k"),
        "top_p": core.config.config_utils.get("model", "default_top_p"),
        "repeat_penalty": core.config.config_utils.get("model", "default_repeat_penalty"),
        "frequency_penalty": core.config.config_utils.get("model", "default_frequency_penalty"),
        "presence_penalty": core.config.config_utils.get("model", "default_presence_penalty"),
        "mirostat": core.config.config_utils.get("model", "default_mirostat"),
        "mirostat_tau": core.config.config_utils.get("model", "default_mirostat_tau"),
        "mirostat_eta": core.config.config_utils.get("model", "default_mirostat_eta"),
    }
    return default_options
