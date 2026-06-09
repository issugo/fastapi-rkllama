"""
Main entry point for the fastapi-rkllama application.

This module initializes the FastAPI application, sets up logging, configures
hardware-specific settings (like NPU frequency), and starts the Uvicorn server.
"""

# Import libs
import logging
import os
import resource
import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import api_router
from core.config import config_utils
from loggers import logging_setup
from loggers.debug_utils import add_debug_api

# Local file

# Check for debug mode using the improved method
DEBUG_MODE = None
logger = None

## app = Flask(__name__)
app = FastAPI()
# Enable CORS for all routes
## CORS(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


# Launch function
def main():
    """
    Main function to launch the FastAPI server.

    It performs the following steps:
    1. Loads settings from configuration.
    2. Sets up logging based on debug mode.
    3. Adds debug APIs if enabled.
    4. Configures NPU frequency for the detected Rockchip processor.
    5. Sets resource limits (max open files).
    6. Starts the Uvicorn server on the configured host and port.
    """
    settings = config_utils.get_settings()
    print(settings)

    # Set debug mode if specified in config - using the improved method
    global DEBUG_MODE
    DEBUG_MODE = settings.is_debug_mode()

    logging_setup(Path(settings.get_path("logs")), DEBUG_MODE)

    # Only include debug endpoint if in debug mode
    if DEBUG_MODE:
        add_debug_api(app=app)

    global logger
    logger = logging.getLogger("rkllama.server")

    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.warning("Debug mode enabled")
        settings.display()
        os.environ["RKLLAMA_DEBUG"] = "1"  # Explicitly set for subprocess consistency

    # Get port from config
    port = settings.server.port

    # Check the processor
    processor = settings.platform.processor.name
    if not processor:
        logger.error("Processor not configured")
        sys.exit(1)
    else:
        if processor not in ["rk3588", "rk3576"]:
            logger.error("Error: Invalid processor. Please enter rk3588 or rk3576.")
            sys.exit(1)
        logger.info(f"Setting the frequency for the {processor} platform...")
        library_path = os.path.join(
            settings.get_path("lib"), f"fix_freq_{processor}.sh"
        )

        # Pass debug flag as parameter to the shell script
        debug_param = "1" if DEBUG_MODE else "0"
        command = f"sudo bash {library_path} {debug_param}"
        subprocess.run(command, shell=True)

    # Set the resource limits
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Start the API server with the chosen port
    logger.info(f"Starting the API at http://localhost:{port}")
    uvicorn.run(
        app,
        host=settings.server.host,
        port=int(port),
        log_level="debug",
    )


if __name__ == "__main__":
    main()


# Modification Summary:
# - Added module-level docstring.
# - Added docstring to the main function for compliance with documentation guidelines.
# - Ensured all app code modifications are documented directly in the code.
