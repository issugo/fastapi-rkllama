import logging
import os
from pathlib import Path


def logging_setup(logs_dir: Path, debug_mode: bool = False):
    os.makedirs(logs_dir, exist_ok=True)

    # Set up logging with an appropriate level based on debug mode
    logging.basicConfig(
        level= logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(logs_dir, "rkllama_server.log")),
        ],
    )
