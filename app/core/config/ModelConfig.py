from typing import Any

import core.config.config_utils
from core import config


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
