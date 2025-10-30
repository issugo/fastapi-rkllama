import datetime
import os
from logging import Logger
from typing import Any, Tuple, List

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from starlette.requests import Request
from starlette.responses import JSONResponse

import core.backends
import core.config
import core.config.config_utils
from api import logger
from core.api.parameters.rkllama_requests import RKPullRequest
from core.model.HfFileInfo import HfFileInfo
from core.model.Model import Model
from core.model.ModelPath import ModelDirException, ModelPath, ModelException
from core.model.ModelType import ModelType
from core.model.storage_helpers.RKPullSupplier import RKPullSupplier

router = APIRouter(tags=["rkllama"])
# Original RKLLAMA Routes:
# POST   /pull
# GET    /models
# POST   /load_model
# POST   /unload_model
# POST   /generate
# DELETE /rm

@router.get("/")
def default_route():
    return JSONResponse(
        {
            "message": "Welcome to FastAPI RKLLama with Ollama API compatibility!",
            "github": "https://github.com/issugo/fastapi-rkllama",
        },
        status_code=200,
    )



@router.post("/pull")
async def pull_model(request: Request, rk_pull_request: RKPullRequest):
    from core.model.storage_helpers.model_pull import pull_model_stream
    from core.model.storage_helpers.SupplierFileInfo import Supplier

    splitted = rk_pull_request.model.split("/")

    class LocalRKPullSupplier(RKPullSupplier):

        @property
        def logger(self) -> Logger:
            return logger

        def check_params(self) -> Any | None:
            if len(splitted) < 3:
                return self.error(f"Invalid path '{rk_pull_request.model}'")
            return None

        def model_type(self, model_name, file, repo) -> Tuple[ModelType | None, Any]:
            if rk_pull_request.model_type is None:
                for mtype in ModelType:
                    if file.endswith(mtype.get_extension()):
                        rk_pull_request.model_type = mtype
                        break

            if rk_pull_request.model_type is None:
                return None, f"Error: Invalid model type '{rk_pull_request.model_type}'\n"

            return rk_pull_request.model_type, None

        def model_data(self) -> Tuple[str, str, str, Supplier]:
            model_name, file, repo = HfFileInfo.model_data(
                split_name=splitted,
                model_name=rk_pull_request.model_name
            )
            repo = rk_pull_request.model.replace(f"/{file}", "")
            return model_name, file, repo, Supplier.HUGGINGFACE


    return pull_model_stream(request=request, pull_supplier=LocalRKPullSupplier())


@router.get("/models")
def list_models():
    try:
        model_list: List[Model] = Model.list()
        return {"models": model_list}
    except ModelDirException as mde:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(mde)}."}),
            status_code=500,
        )

@router.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str):
    try:
        model:Model = Model.load(model_path=ModelPath.from_model_id(model_id))
        return model
    except ModelException as me:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(me)}."}),
            status_code=500,
        )




@router.post("/generate")
async def recevoir_message(request: Request):
    from core.processing.process import rkllm_request
    from core.processing import rkllama_handler
    from core.backends.GlobalState import GLOBAL_STATE

    if not GLOBAL_STATE.backend:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    GLOBAL_STATE.backend.usage_lock.acquire()
    logger.info("Processing generate request")

    # Use custom_request if provided, otherwise use Flask's request
    # req = custom_request if custom_request is not None else request
    # data = req.json
    data = await request.json()

    return await rkllm_request(
        rkllm_model=GLOBAL_STATE.backend.backend,
        model_shared_data=GLOBAL_STATE.backend.shared_data,
        model_file=GLOBAL_STATE.backend.model_file,
        usage_lock=GLOBAL_STATE.backend.usage_lock,
        handler=rkllama_handler,
        data=data)


@router.post("/unload_model")
def unload_model_route():
    from core.backends.GlobalState import GLOBAL_STATE
    if not GLOBAL_STATE.backend:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    GLOBAL_STATE.backend.unload()
    GLOBAL_STATE.backend = None

    return JSONResponse({"message": "Model successfully unloaded!"}, status_code=200)


@router.post("/load_model")
async def load_model_route(request: Request):
    from core.model.ModelFile import ModelFile, ModelFileInfo
    from core.backends.GlobalState import GLOBAL_STATE

    # Check if a model is currently loaded
    if GLOBAL_STATE.backend:
        return JSONResponse(
            jsonable_encoder({"error": f"model {GLOBAL_STATE.backend.model_file.model_name} is already loaded. Please unload it first."}),
            status_code=400,
        )

    data = await request.json()
    if "model_name" not in data:
        return JSONResponse(
            jsonable_encoder({"error": "Please enter the name of the model to be loaded."}),
            status_code=400,
        )

    model_file_info: ModelFileInfo = ModelFileInfo(**data)
    model_file: ModelFile = ModelFile.create(model_file_info, )
    GLOBAL_STATE.backend, error = model_file.load_model()

    if error:
        return JSONResponse({"error": error}, status_code=400)
    elif GLOBAL_STATE.backend:
        return JSONResponse(
            {"message": f"Model {model_file.model_name} loaded successfully."}, status_code=200
        )
    else:
        return JSONResponse({"error": f"unexpected error loading Model {model_file.model_name}"}, status_code=400)



@router.delete("/rm")
async def Rm_model(request: Request):
    data = await request.json()
    if "model" not in data:
        return JSONResponse({"error": "Please specify a model."}, status_code=400)

    model_path = os.path.join(core.config.config_utils.get_path("models"), data["model"])
    if not os.path.exists(model_path):
        return JSONResponse(
            {"error": f"The model: {data['model']} cannot be found."}, status_code=404
        )

    os.remove(model_path)

    return JSONResponse(
        {"message": "The model has been successfully deleted!"}, status_code=200
    )




@router.get("/current_models")
def get_current_models():
    from core.model import Model, ModelMetadata
    from core.model.ModelInfo import ModelInfo
    from core.model.ModelPath import ModelType

    # Get the models info from Modelfile and HF
    models_dir = core.config.config_utils.get_path("models")
    models_info = {}
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                for mtype in ModelType:
                    if file.endswith(mtype.get_extension()):
                        size = ModelMetadata.get_model_size(os.path.join(subdir_path, file))

                        # Extract parameter size and quantization details if available
                        model_details = Model.extract_model_details(model_type=mtype, model_name=file)

                        models_info[subdir] = ModelInfo(
                            name=subdir,  # Use simplified name like qwen:3b
                            model=subdir,  # Match Ollama's format
                            modified_at=datetime.datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(subdir_path, file))
                            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            size=size,
                            digest="",  # Ollama field (not used but included for compatibility)
                            details=model_details,
                            model_type=mtype
                        )
                        break

    # Loop over the models currently running
    models_running = []
    for model in variables.worker_manager_rkllm.workers.keys():
        worker_model_info = variables.worker_manager_rkllm.workers[model].worker_model_info
        model_info = {
            "name": model,
            "model": model,
            "size": worker_model_info.size,
            "digest": models_info[model]["digest"],
            "details": {
                "parent_model": "",
                "format": models_info[model]["details"]["format"],
                "family": models_info[model]["details"]["family"],
                "families": [
                    models_info[model]["details"]["family"]
                ],
                "parameter_size": models_info[model]["details"]["parameter_size"],
                "quantization_level": models_info[model]["details"]["quantization_level"]
            },
            "expires_at": worker_model_info.expires_at.strftime('%Y-%m-%d %H:%M:%S.%f'),
            "loaded_at": worker_model_info.loaded_at.strftime('%Y-%m-%d %H:%M:%S.%f'),
            "base_domain_id": worker_model_info.base_domain_id,
            "last_call": worker_model_info.last_call.strftime('%Y-%m-%d %H:%M:%S.%f')
        }
        models_running.append(model_info)

    return jsonify({"models": models_running}), 200


@router.post("/api/create")
async def create_model(request: Request):
    data = await request.json()
    model_name = core.config.config_utils.get("name")
    modelfile = core.config.config_utils.get("modelfile", "")

    if DEBUG_MODE:
        logger.debug(f"API create request data: {data}")

    if not model_name:
        return JSONResponse({"error": "Missing model name"}, status_code=400)

    model_dir = os.path.join(core.config.config_utils.get_path("models"), model_name)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "Modelfile"), "w") as f:
        f.write(modelfile)

    # Parse the modelfile to extract parameters
    modelfile_lines = modelfile.strip().split("\n")
    from_line = next(
        (line for line in modelfile_lines if line.startswith("FROM=")), None
    )
    huggingface_path = next(
        (line for line in modelfile_lines if line.startswith("HUGGINGFACE_PATH=")), None
    )

    if not from_line or not huggingface_path:
        return JSONResponse(
            {"error": "Invalid Modelfile: missing FROM or HUGGINGFACE_PATH"},
            status_code=400,
        )

    # Extract values
    from_value = from_line.split("=")[1].strip("\"'")
    huggingface_path = huggingface_path.split("=")[1].strip("\"'")

    # For compatibility with existing implementation
    return JSONResponse({"status": "success", "model": model_name}, status_code=200)


