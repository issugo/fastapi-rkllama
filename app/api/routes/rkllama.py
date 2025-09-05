from fastapi import APIRouter
from starlette.responses import JSONResponse

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
