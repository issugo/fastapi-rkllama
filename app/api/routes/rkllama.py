from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.processing import rkllama_handler
from core.endpoints.rkllm import GLOBAL_STATE
from core.model.ModelFile import ModelFile, ModelFileInfo
from core.processing.process import rkllm_request
from main import logger

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
    if not GLOBAL_STATE.rkllm_model:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    GLOBAL_STATE.rkllm_model.usage_lock.acquire()
    logger.info("Processing generate request")

    # Use custom_request if provided, otherwise use Flask's request
    # req = custom_request if custom_request is not None else request
    # data = req.json
    data = await request.json()

    return await rkllm_request(
        rkllm_model=GLOBAL_STATE.rkllm_model.rkllm_model,
        model_shared_data=GLOBAL_STATE.rkllm_model.shared_data,
        model_file=GLOBAL_STATE.rkllm_model.model_file,
        usage_lock=GLOBAL_STATE.rkllm_model.usage_lock,
        handler=rkllama_handler,
        data=data)


@router.post("/unload_model")
def unload_model_route():
    if not GLOBAL_STATE.rkllm_model:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    GLOBAL_STATE.rkllm_model.unload()
    GLOBAL_STATE.rkllm_model = None

    return JSONResponse({"message": "Model successfully unloaded!"}, status_code=200)


@router.post("/load_model")
async def load_model_route(request: Request):
    # Check if a model is currently loaded
    if GLOBAL_STATE.rkllm_model:
        return JSONResponse(
            {"error": f"model {GLOBAL_STATE.rkllm_model.model_file.model_name} is already loaded. Please unload it first."},
            status_code=400,
        )

    data = await request.json()
    if "model_name" not in data:
        return JSONResponse(
            {"error": "Please enter the name of the model to be loaded."},
            status_code=400,
        )

    model_file_info: ModelFileInfo = ModelFileInfo(**data)
    model_file: ModelFile = ModelFile.create_model(model_file_info)
    GLOBAL_STATE.rkllm_model, error = model_file.load_model()

    if error:
        return JSONResponse({"error": error}, status_code=400)
    elif GLOBAL_STATE.rkllm_model:
        return JSONResponse(
            {"message": f"Model {model_file.model_name} loaded successfully."}, status_code=200
        )
    else:
        return JSONResponse({"error": f"unexpected error loading Model {model_file.model_name}"}, status_code=400)
