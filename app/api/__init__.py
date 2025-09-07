from fastapi import APIRouter
import logging

# Set up logger for this package
logger = logging.getLogger("api")
from app.api.routes import rkllama, ollama

api_router = APIRouter()
api_router.include_router(rkllama.router)
api_router.include_router(ollama.router)
