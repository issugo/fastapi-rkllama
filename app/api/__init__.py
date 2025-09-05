from fastapi import APIRouter

from app.api.routes import rkllama, ollama

api_router = APIRouter()
api_router.include_router(rkllama.router)
api_router.include_router(ollama.router)
