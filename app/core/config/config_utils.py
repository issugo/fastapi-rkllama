from functools import lru_cache

from core.config.RKLLAMAConfig import RKLLAMAConfig, RKLLAMASettings

rkllama_config: RKLLAMAConfig | None = None

@lru_cache
def get_settings() -> RKLLAMASettings:
    return RKLLAMASettings()

