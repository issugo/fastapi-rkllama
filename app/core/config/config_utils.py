from typing import Any
from functools import lru_cache

from core.config.RKLLAMAConfig import RKLLAMAConfig, RKLLAMASettings

rkllama_config: RKLLAMAConfig | None = None


@lru_cache
def get_settings() -> RKLLAMASettings:
    return RKLLAMASettings()


def get_path(key: str, default: Any = None) -> str:
    return get_settings().get_path(key, default)


def get(key: str, default: Any = None) -> Any:
    settings = get_settings()
    if hasattr(settings, key):
        return getattr(settings, key)
    for section in [settings.model, settings.paths, settings.server, settings.platform]:
        if hasattr(section, key):
            return getattr(section, key)
    if key == "name":
        return settings.model.default or default
    try:
        config_dict = settings.model_dump()
        if key in config_dict:
            return config_dict[key]
        for section_name, section_val in config_dict.items():
            if isinstance(section_val, dict) and key in section_val:
                return section_val[key]
    except Exception:
        pass
    return default
