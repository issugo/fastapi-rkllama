import logging

# Set up logger for this package
logger = logging.getLogger("core.model")

current_model = None  # Global variable for storing the loaded model
rkllm_model = None  # Model instance
