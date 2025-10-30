import logging

from fastapi import APIRouter

# Set up logger for this package
logger = logging.getLogger("api")

from api.routes import rkllama, ollama_new, openai_new
from api.routes.converter import rkllm_converter

api_router = APIRouter()
api_router.include_router(rkllama.router)
api_router.include_router(ollama_new.router)
api_router.include_router(openai_new.router)

api_router.include_router(rkllm_converter.router)
