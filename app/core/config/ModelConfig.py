from typing import Any

from core import config


def get_model_default_options() -> dict[str | Any, Any]:
    default_options = {
        "temperature": config.get("model", "default_temperature"),
        "num_ctx": config.get("model", "default_num_ctx"),
        "max_new_tokens": config.get("model", "default_max_new_tokens"),
        "top_k": config.get("model", "default_top_k"),
        "top_p": config.get("model", "default_top_p"),
        "repeat_penalty": config.get("model", "default_repeat_penalty"),
        "frequency_penalty": config.get("model", "default_frequency_penalty"),
        "presence_penalty": config.get("model", "default_presence_penalty"),
        "mirostat": config.get("model", "default_mirostat"),
        "mirostat_tau": config.get("model", "default_mirostat_tau"),
        "mirostat_eta": config.get("model", "default_mirostat_eta"),
    }
    return default_options
