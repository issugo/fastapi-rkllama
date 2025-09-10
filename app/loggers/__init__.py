import logging
import os

import core.config.config_utils
from core.config.config_utils import is_debug_mode

DEBUG_MODE = is_debug_mode()
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO


def setup():
    # Ensure logs directory exists before configuring logging
    logs_dir = core.config.config_utils.get_path("logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Set up logging with appropriate level based on debug mode
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(logs_dir, "rkllama_server.log")),
        ],
    )
