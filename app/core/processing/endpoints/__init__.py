import logging

# Set up logger for this package
logger = logging.getLogger("core.processing.endpoints")

from .EndpointHandler import EndpointHandler
from .ChatEndpointHandler import ChatEndpointHandler
from .GenerateEndpointHandler import GenerateEndpointHandler
