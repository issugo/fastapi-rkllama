import logging
from typing import Optional

from core.processing.APIHandler import APIHandler

# Set up logger for this package
logger = logging.getLogger("core.processing")

ollama_handler: Optional[APIHandler] = None
openai_handler: Optional[APIHandler] = None
rkllama_handler: Optional[APIHandler] = None
