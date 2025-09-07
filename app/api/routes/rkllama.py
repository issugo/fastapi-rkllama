import os

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

from core import model
from core.processing.process import CustomRequest
from main import DEBUG_MODE, logger

router = APIRouter(tags=["rkllama"])


@router.get("/")
def default_route():
    return JSONResponse(
        {
            "message": "Welcome to RKLLama with Ollama API compatibility!",
            "github": "https://github.com/notpunhnox/rkllama",
        },
        status_code=200,
    )


@router.post("/generate")
async def recevoir_message(request: Request):
    if not model.rkllm_model:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    # define modelfile path
    modelfile = os.path.join(model.rkllm_model.model_dir, "Modelfile")

    # variables.verrou.acquire()
    logger.info("Processing generate request")
    return await CustomRequest(model.rkllm_model, modelfile, request)
