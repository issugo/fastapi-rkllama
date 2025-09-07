import logging
from typing import Union, Optional

from core.processing import Handler
from core.processing.Handler import Handler

# Set up logger for this package
logger = logging.getLogger("core.processing")

ollama_handler: Optional[Handler|None] = None
openai_handler: Optional[Handler|None] = None
rkllama_handler: Optional[Handler|None] = None
