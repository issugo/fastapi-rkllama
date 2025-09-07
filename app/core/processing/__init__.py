import logging
from typing import Union, Optional

from core.processing import Handler
from core.processing.Handler import APIHandler

# Set up logger for this package
logger = logging.getLogger("core.processing")

ollama_handler: Optional[APIHandler | None] = None
openai_handler: Optional[APIHandler | None] = None
rkllama_handler: Optional[APIHandler | None] = None
