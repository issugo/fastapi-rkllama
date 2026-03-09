# Class to manage the information for running RKLLM models
import logging
from datetime import datetime, timedelta

from core.config.RKLLAMAConfig import RKLLAMASettings
from core.model.ModelFile import ModelFile
from core.processing.BaseDomainId import BaseDomainId

logger = logging.getLogger("rkllama.worker")

setting: RKLLAMASettings | None = None
DEBUG_MODE: bool | None = None


class WorkerModelInfo:
    def __init__(self, modelfile: ModelFile, base_domain_id: BaseDomainId):

        global settings
        if settings is None:
            from core.config import config_utils
            settings = config_utils.get_settings()

        global DEBUG_MODE
        if DEBUG_MODE is None:
            DEBUG_MODE = settings.is_debug_mode()

        if DEBUG_MODE:
            logger.debug(f"new WorkerModelInfo")

        self.modelfile = modelfile
        # config.get("modelfile", "max_minutes_loaded_in_memory")
        self.expires_at = datetime.now() + timedelta(minutes=settings.server.max_minutes_loaded_in_memory)
        self.loaded_at = datetime.now()
        self.base_domain_id = base_domain_id
        self.last_call = datetime.now()

    @property
    def size(self) -> int:
        return self.modelfile.model.size
