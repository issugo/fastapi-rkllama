import datetime
import os
import shutil
import urllib.parse

import requests
from fastapi import APIRouter
from huggingface_hub import HfFileSystem, hf_hub_url
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

import core.backends
from core import config
from core.model import Model, ModelMetadata
from core.model.ModelInfo import ModelInfo
from core.model.ModelPath import ModelType
from core.processing import rkllama_handler
from core.backends.GlobalState import GLOBAL_STATE
from core.model.ModelFile import ModelFile, ModelFileInfo
from core.processing.process import rkllm_request
from main import logger, DEBUG_MODE

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
    if not GLOBAL_STATE.endpoint:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    GLOBAL_STATE.endpoint.usage_lock.acquire()
    logger.info("Processing generate request")

    # Use custom_request if provided, otherwise use Flask's request
    # req = custom_request if custom_request is not None else request
    # data = req.json
    data = await request.json()

    return await rkllm_request(
        rkllm_model=GLOBAL_STATE.endpoint.endpoint,
        model_shared_data=GLOBAL_STATE.endpoint.shared_data,
        model_file=GLOBAL_STATE.endpoint.model_file,
        usage_lock=GLOBAL_STATE.endpoint.usage_lock,
        handler=rkllama_handler,
        data=data)


@router.post("/unload_model")
def unload_model_route():
    if not GLOBAL_STATE.endpoint:
        return JSONResponse(
            {"error": "No models are currently loaded."}, status_code=400
        )

    GLOBAL_STATE.endpoint.unload()
    GLOBAL_STATE.endpoint = None

    return JSONResponse({"message": "Model successfully unloaded!"}, status_code=200)


@router.post("/load_model")
async def load_model_route(request: Request):
    # Check if a model is currently loaded
    if GLOBAL_STATE.endpoint:
        return JSONResponse(
            {"error": f"model {GLOBAL_STATE.endpoint.model_file.model_name} is already loaded. Please unload it first."},
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
    GLOBAL_STATE.endpoint, error = model_file.load_model()

    if error:
        return JSONResponse({"error": error}, status_code=400)
    elif GLOBAL_STATE.endpoint:
        return JSONResponse(
            {"message": f"Model {model_file.model_name} loaded successfully."}, status_code=200
        )
    else:
        return JSONResponse({"error": f"unexpected error loading Model {model_file.model_name}"}, status_code=400)


@router.get("/models")
def list_models():
    # Return the list of available models using config path
    models_dir = config.get_path("models")

    if not os.path.exists(models_dir):
        return JSONResponse(
            {"error": f"The models directory {models_dir} is not found."},
            status_code=500,
        )

    direct_models = [f for f in os.listdir(models_dir) if f.endswith(".rkllm")]

    for model in direct_models:
        model_name = os.path.splitext(model)[0]
        model_dir = os.path.join(models_dir, model_name)

        os.makedirs(model_dir, exist_ok=True)

        shutil.move(os.path.join(models_dir, model), os.path.join(model_dir, model))

    model_dirs = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    model_dirs.append(subdir)
                    break

    return JSONResponse({"models": model_dirs}, status_code=200)


@router.delete("/rm")
async def Rm_model(request: Request):
    data = await request.json()
    if "model" not in data:
        return JSONResponse({"error": "Please specify a model."}, status_code=400)

    model_path = os.path.join(config.get_path("models"), data["model"])
    if not os.path.exists(model_path):
        return JSONResponse(
            {"error": f"The model: {data['model']} cannot be found."}, status_code=404
        )

    os.remove(model_path)

    return JSONResponse(
        {"message": "The model has been successfully deleted!"}, status_code=200
    )


@router.post("/pull")
async def pull_model(request: Request):
    data = await request.json()

    ## @stream_with_context
    async def generate_progress():
        if "model" not in data:
            yield "Error: Model not specified.\n"
            return

        splitted = data["model"].split("/")
        model_name = splitted[1] if "model_name" not in data else data["model_name"]
        if len(splitted) < 3:
            yield f"Error: Invalid path '{data['model']}'\n"
            return

        file = splitted[2]
        repo = data["model"].replace(f"/{file}", "")

        try:
            # Use Hugging Face HfFileSystem to get the file metadata
            fs = HfFileSystem()
            file_info = fs.info(repo + "/" + file)

            total_size = file_info["size"]  # File size in bytes
            if total_size == 0:
                yield "Error: Unable to retrieve file size.\n"
                return

            # Use config to get models path
            # model_dir = os.path.join(config.get_path("models"), file.replace('.rkllm', ''))
            model_dir = os.path.join(config.get_path("models"), model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Define a file to download
            local_filename = os.path.join(model_dir, file)

            # Create fonfiguration file for model
            #ModelFile.create_model(ModelFileInfo(huggingface_path=repo, model_file=file, model_name=model_name))
            model_file: ModelFile = ModelFile.create(ModelFileInfo(huggingface_path=repo, endpoint_model_file=file, model_name=model_name))

            yield f"Downloading {file} ({total_size / (1024**2):.2f} MB)...\n"

            try:
                # Download the file with progress
                url = hf_hub_url(repo_id=repo, filename=file)
                with (
                    requests.get(url, stream=True) as r,
                    open(local_filename, "wb") as f,
                ):
                    downloaded_size = 0
                    chunk_size = 8192  # 8KB

                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            progress = int((downloaded_size / total_size) * 100)
                            yield f"{progress}%\n"

            except Exception as download_error:
                # Remove the file if an error occurs during download
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                yield f"Error during download: {str(download_error)}\n"
                return

        except Exception as e:
            yield f"Error: {str(e)}\n"

    # Use the appropriate content type for streaming responses
    ## is_ollama_request = request.path.startswith('/api/')
    logger.debug(f"request.url={request.url}")
    urlparsed_path = urllib.parse.urlparse(str(request.url)).path
    logger.debug(f"urlparsed_path={urlparsed_path}")
    is_ollama_request = urlparsed_path.startswith("/api/")
    content_type = "application/x-ndjson" if is_ollama_request else "text/plain"
    return StreamingResponse(
        generate_progress(), headers={"Content-Type": content_type}
    )


@router.get("/current_models")
def get_current_models():
    # Get the models info from Modelfile and HF
    models_dir = config.get_path("models")
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
    model_name = config.get("name")
    modelfile = config.get("modelfile", "")

    if DEBUG_MODE:
        logger.debug(f"API create request data: {data}")

    if not model_name:
        return JSONResponse({"error": "Missing model name"}, status_code=400)

    model_dir = os.path.join(config.get_path("models"), model_name)
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
