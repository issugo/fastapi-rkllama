import datetime
import json
import os
import re
import shutil
import urllib.parse

import requests
from fastapi import APIRouter
from huggingface_hub import HfFileSystem, hf_hub_url
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

import core.backends
import core.config
import core.config.config_utils
from api import logger
from core.api.parameters.rkllama_requests import RKPullRequest
from core.model.ModelPath import find_rkllm_model_name, get_huggingface_model_info
from core.model.ModelType import ModelType

router = APIRouter(tags=["rkllama"])
# Original RKLLAMA Routes:
# GET    /models
# POST   /load_model
# POST   /unload_model
# POST   /generate
# POST   /pull
# DELETE /rm

@router.post("/pull")
async def pull_model(request: Request, rk_pull_request: RKPullRequest):
    from core.model.ModelFile import ModelFile, ModelFileInfo
    from core.config import config_utils

    async def generate_progress():
        splitted = rk_pull_request.model.split("/")
        if len(splitted) < 3:
            yield f"Error: Invalid path '{rk_pull_request.model}'\n"
            return

        model_name = splitted[1] if rk_pull_request.model_name is None else rk_pull_request.model_name
        file = splitted[2]
        repo = rk_pull_request.model.replace(f"/{file}", "")

        logger.debug(f"model_name={model_name}, file={file}, repo={repo}")

        if rk_pull_request.model_type is None:
            for mtype in ModelType:
                if file.endswith(mtype.get_extension()):
                    rk_pull_request.model_type = mtype
                    break

        if rk_pull_request.model_type is None:
            yield f"Error: Invalid model type '{rk_pull_request.model_type}'\n"
            return

        try:
            # Use Hugging Face HfFileSystem to get the file metadata
            fs = HfFileSystem()
            file_info = fs.info(repo + "/" + file)

            logger.debug(f"file_info={file_info}")

            total_size = file_info["size"]  # File size in bytes
            if total_size == 0:
                yield "Error: Unable to retrieve file size.\n"
                return

            # Create the configuration file for model
            model_type = \
                rk_pull_request.model_type if rk_pull_request.model_type is not None \
                else ModelType.get_model_type_from_endpoint_model_file(file)
            logger.debug(f"{model_type}")
            model_file_info: ModelFileInfo = ModelFileInfo(
                model_name=model_name,
                model_type=model_type,
                huggingface_path=repo,
                endpoint_model_file=file,
                endpoint_model_file_size=total_size,
            )
            logger.debug(f"{model_file_info.model_dump_json()}")
            model_file: ModelFile = ModelFile.create(
                model_file_info=model_file_info,
                default_model_config=config_utils.rkllama_config.model )
            logger.debug(f"{model_file.model_dump_json()}")

            logger.debug(f"{model_file.simple_model_metadata.model_dump_json()}")

            if model_file.is_locked():
                yield "Error: Model is currently locked.\n"
                return

            lock_id = model_file.lock_model()
            if lock_id > 0:

                # Define a file to download
                local_filename = model_file.endpoint_model_file_path

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

                    model_file.unlock_model(lock_id)

                except Exception as download_error:
                    # Remove the file if an error occurs during download
                    if os.path.exists(local_filename):
                        os.remove(local_filename)
                    yield f"Error during download: {str(download_error)}\n"
                    model_file.unlock_model(lock_id)
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
            {"error": f"model {GLOBAL_STATE.backend.model_file.model_name} is already loaded. Please unload it first."},
            status_code=400,
        )

    data = await request.json()
    if "model_name" not in data:
        return JSONResponse(
            {"error": "Please enter the name of the model to be loaded."},
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


@router.get("/models")
def list_models():
    # Return the list of available models using config path
    models_dir = core.config.config_utils.get_path("models")

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


@router.post("/api/show")
def show_model_info(request: Request):
    ##### Github Copilot Start Workaround
    request_data = request.get_data().decode("UTF-8")
    request_data = request_data.replace("'", '"')
    data = json.loads(request_data) if request_data else {}
    model_name = data.get("name") if "name" in data else data.get("model")
    ##### # Github Copilot End

    if DEBUG_MODE:
        logger.debug(f"API show request data: {data}")

    if not model_name:
        return JSONResponse({"error": "Missing model name"}, status_code=400)

    model_dir = os.path.join(core.config.config_utils.get_path("models"), model_name)
    model_rkllm = find_rkllm_model_name(model_dir)

    if not os.path.exists(model_dir):
        return JSONResponse(
            {"error": f"Model '{model_name}' not found"}, status_code=404
        )

    # Read modelfile content if available
    modelfile_path = os.path.join(model_dir, "Modelfile")
    modelfile_content = ""
    system_prompt = ""
    template = "{{ .Prompt }}"
    license_text = ""
    huggingface_path = None

    if os.path.exists(modelfile_path):
        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()

            # Extract system prompt if available
            system_match = re.search(r'SYSTEM="(.*?)"', modelfile_content, re.DOTALL)
            if system_match:
                system_prompt = system_match.group(1).strip()

            # Check for template pattern
            template_match = re.search(
                r'TEMPLATE="(.*?)"', modelfile_content, re.DOTALL
            )
            if template_match:
                template = template_match.group(1).strip()

            # Check for LICENSE pattern (some modelfiles have this)
            license_match = re.search(r'LICENSE="(.*?)"', modelfile_content, re.DOTALL)
            if license_match:
                license_text = license_match.group(1).strip()

            # Extract HuggingFace path for API access
            hf_path_match = re.search(
                r'HUGGINGFACE_PATH="(.*?)"', modelfile_content, re.DOTALL
            )
            if hf_path_match:
                huggingface_path = hf_path_match.group(1).strip()

            # Extract temperature if available
            temp_match = re.search(r"TEMPERATURE=(\d+\.?\d*)", modelfile_content)
            if temp_match:
                try:
                    temperature = float(temp_match.group(1))
                except ValueError:
                    pass

    # Find the .rkllm file
    model_file = None
    for file in os.listdir(model_dir):
        if file.endswith(".rkllm"):
            model_file = file
            break

    if not model_file:
        return JSONResponse(
            {"error": f"Model file not found in '{model_name}' directory"},
            status_code=404,
        )

    file_path = os.path.join(model_dir, model_file)
    size = os.path.getsize(file_path)

    # Extract model details
    model_details = extract_model_details(model_rkllm)
    parameter_size = core.config.config_utils.get("parameter_size", "Unknown")
    quantization_level = core.config.config_utils.get("quantization_level", "Unknown")

    # Determine model family based on name patterns
    family = "llama"  # default family
    families = ["llama"]

    # Try to get enhanced information from Hugging Face API
    hf_metadata = (
        get_huggingface_model_info(huggingface_path) if huggingface_path else None
    )

    # Use HF metadata to improve model info if available
    if hf_metadata:
        # Extract tags from HF metadata
        tags = hf_metadata.get("tags", [])

        # Better determine model family based on HF tags or architecture field
        if (
                hf_metadata.get("architecture") == "qwen"
                or "qwen" in tags
                or "qwen2" in tags
        ):
            family = "qwen2"
            families = ["qwen2"]
        elif hf_metadata.get("architecture") == "mistral" or "mistral" in tags:
            family = "mistral"
            families = ["mistral"]
        elif hf_metadata.get("architecture") == "deepseek" or "deepseek" in tags:
            family = "deepseek"
            families = ["deepseek"]
        elif hf_metadata.get("architecture") == "phi" or "phi" in tags:
            family = "phi"
            families = ["phi"]
        elif hf_metadata.get("architecture") == "gemma" or "gemma" in tags:
            family = "gemma"
            families = ["gemma"]
        elif "tinyllama" in tags:
            family = "tinyllama"
            families = ["tinyllama", "llama"]
        elif any("llama-3" in tag for tag in tags) or any(
                "llama3" in tag for tag in tags
        ):
            family = "llama3"
            families = ["llama3", "llama"]
        elif any("llama-2" in tag for tag in tags) or any(
                "llama2" in tag for tag in tags
        ):
            family = "llama2"
            families = ["llama2", "llama"]

        # Extract model card metadata
        model_card = hf_metadata.get("cardData", {})

        # Better parameter size from HF metadata
        parameter_count = None
        if "params" in model_card:
            try:
                params = int(model_card["params"])
                if params >= 1_000_000_000:
                    parameter_size = f"{params / 1_000_000_000:.1f}B".replace(
                        ".0B", "B"
                    )
                    # Also store the raw parameter count for model_info
                    parameter_count = params
            except (ValueError, TypeError):
                parameter_count = None
        else:
            parameter_count = None

        # Extract quantization info
        if "quantization" in hf_metadata:
            quantization_level = hf_metadata["quantization"]

        # Better license information
        if "license" in hf_metadata and not license_text:
            license_text = hf_metadata["license"]
    else:
        # Fallback to pattern matching if no HF metadata
        if re.search(r"(?i)Qwen", model_name):
            family = "qwen2"
            families = ["qwen2"]
        elif re.search(r"(?i)Mistral", model_name):
            family = "mistral"
            families = ["mistral"]
        elif re.search(r"(?i)DeepSeek", model_name):
            family = "deepseek"
            families = ["deepseek"]
        elif re.search(r"(?i)Phi", model_name):
            family = "phi"
            families = ["phi"]
        elif re.search(r"(?i)Gemma", model_name):
            family = "gemma"
            families = ["gemma"]
        elif re.search(r"(?i)TinyLlama", model_name):
            family = "tinyllama"
            families = ["tinyllama", "llama"]
        elif re.search(r"(?i)Llama[-_]?3", model_name):
            family = "llama3"
            families = ["llama3", "llama"]
        elif re.search(r"(?i)Llama[-_]?2", model_name):
            family = "llama2"
            families = ["llama2", "llama"]

        parameter_count = None

    # Convert modelfile to Ollama-compatible format
    ollama_modelfile = '# Modelfile generated by "ollama show"\n'
    ollama_modelfile += "# To build a new Modelfile based on this, replace FROM with:\n"
    ollama_modelfile += f"# FROM {model_name}\n\n"

    # Change this section to use a more compatible FROM format
    # Instead of absolute paths, use the model file name which is more compatible with Ollama
    # Original: model_blob_path = f"{model_dir}/{model_file}"

    if DEBUG_MODE:
        # In debug mode, use absolute paths to help with troubleshooting
        model_blob_path = f"{model_dir}/{model_file}"
        ollama_modelfile += f"FROM {model_blob_path}\n"
    else:
        # In normal mode, use the simplified name format that Ollama clients expect
        ollama_modelfile += f"FROM {model_name}\n"

    if template != "{{ .Prompt }}":
        ollama_modelfile += f'TEMPLATE """{template}"""\n'

    if system_prompt:
        ollama_modelfile += f'SYSTEM "{system_prompt}"\n'

    if license_text:
        ollama_modelfile += f'LICENSE """{license_text}"""\n'

    # Additional model info from HF
    model_description = ""
    repo_url = None
    if hf_metadata:
        model_description = hf_metadata.get("description", "").strip()

        # Add description comment to modelfile if available
        if model_description:
            desc_lines = model_description.split("\n")
            desc_comment = "\n".join(
                [f"# {line}" for line in desc_lines[:5]]
            )  # First 5 lines only
            ollama_modelfile = desc_comment + "\n\n" + ollama_modelfile

        # Extract repo URL if available
        if huggingface_path:
            repo_url = f"https://huggingface.co/{huggingface_path}"

    # Parse parameter size into numeric format
    numeric_param_size = None
    if parameter_size != "Unknown":
        param_match = re.search(r"(\d+\.?\d*)B", parameter_size)
        if param_match:
            try:
                size_in_billions = float(param_match.group(1))
                numeric_param_size = int(size_in_billions * 1_000_000_000)
            except ValueError:
                pass

    # Use parameter_count from HF metadata if available, otherwise use parsed value
    if parameter_count is None and numeric_param_size is not None:
        parameter_count = numeric_param_size
    elif parameter_count is None:
        # Default fallback
        if "7B" in model_name or "7b" in model_name:
            parameter_count = 7000000000
        elif "3B" in model_name or "3b" in model_name:
            parameter_count = 3000000000
        elif "1.5B" in model_name or "1.5b" in model_name:
            parameter_count = 1500000000
        else:
            parameter_count = 0

    # Extract base model name (without fine-tuning suffixes)
    base_name = model_name.split("-")[0]

    # Determine finetune type if present
    finetune = None
    if "instruct" in model_name.lower():
        finetune = "Instruct"
    elif "chat" in model_name.lower():
        finetune = "Chat"

    # Build a more complete model_info dict with architecture details
    model_info = {
        "general.architecture": family,
        "general.base_model.0.name": f"{base_name} {parameter_size}",
        "general.base_model.0.organization": family.capitalize(),
        "general.basename": base_name,
        "general.file_type": 15,  # RKLLM file type
        "general.parameter_count": parameter_count,
        "general.quantization_version": 2,
        "general.size_label": parameter_size,
        "general.tags": ["chat", "text-generation"],
        "general.type": "model",
        "tokenizer.ggml.pre": family,
    }

    # Add repo URL if available
    if repo_url:
        model_info["general.base_model.0.repo_url"] = repo_url
        model_info["general.base_model.count"] = 1

    # Add finetune info if available
    if finetune:
        model_info["general.finetune"] = finetune

    # Add license info if available
    if license_text:
        license_name = "other"
        license_link = ""

        # Try to detect common licenses
        if "apache" in license_text.lower():
            license_name = "apache-2.0"
        elif "mit" in license_text.lower():
            license_name = "mit"
        elif "qwen research" in license_text.lower():
            license_name = "qwen-research"

        if huggingface_path:
            license_link = (
                f"https://huggingface.co/{huggingface_path}/blob/main/LICENSE"
            )

        model_info["general.license"] = license_name
        if license_link:
            model_info["general.license.link"] = license_link
        model_info["general.license.name"] = license_name

    # Add language info if we can detect it
    if hf_metadata and "languages" in hf_metadata:
        model_info["general.languages"] = hf_metadata["languages"]
    else:
        # Default to English
        model_info["general.languages"] = ["en"]

    # Add architecture-specific parameters based on model family
    if family == "qwen2":
        model_info.update(
            {
                "qwen2.attention.head_count": 16,
                "qwen2.attention.head_count_kv": 2,
                "qwen2.attention.layer_norm_rms_epsilon": 0.000001,
                "qwen2.block_count": 36 if "3B" in parameter_size else 24,
                "qwen2.context_length": 32768,
                "qwen2.embedding_length": 2048 if "3B" in parameter_size else 1536,
                "qwen2.feed_forward_length": 11008 if "3B" in parameter_size else 8192,
                "qwen2.rope.freq_base": 1000000,
            }
        )
    elif family == "llama" or family == "llama2" or family == "llama3":
        model_info.update(
            {
                f"{family}.attention.head_count": 32,
                f"{family}.attention.head_count_kv": 4,
                f"{family}.attention.layer_norm_rms_epsilon": 0.000001,
                f"{family}.block_count": 32,
                f"{family}.context_length": 4096,
                f"{family}.embedding_length": 4096,
                f"{family}.feed_forward_length": 11008,
                f"{family}.rope.freq_base": 10000,
            }
        )
    elif family == "mistral":
        model_info.update(
            {
                "mistral.attention.head_count": 32,
                "mistral.attention.head_count_kv": 8,
                "mistral.attention.layer_norm_rms_epsilon": 0.000001,
                "mistral.block_count": 32,
                "mistral.context_length": 8192,
                "mistral.embedding_length": 4096,
                "mistral.feed_forward_length": 14336,
            }
        )

    # Calculate modified timestamp
    modified_at = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    # Format parameters string nicely
    parameters_str = parameter_size
    if parameters_str == "Unknown" and parameter_count:
        if parameter_count >= 1_000_000_000:
            parameters_str = f"{parameter_count / 1_000_000_000:.1f}B".replace(
                ".0B", "B"
            )
        else:
            parameters_str = f"{parameter_count / 1_000_000:.1f}M".replace(".0M", "M")

    # Capabilities based on model family. ### Github Copilot requires this
    capabilities = ["completion"]
    if family in ["qwen2", "phi", "llama3", "mistral"]:
        capabilities.append("tools")

    # Prepare response with enhanced metadata
    response = {
        "license": license_text or "Unknown",
        "modelfile": ollama_modelfile,
        "parameters": parameters_str,
        "template": template,
        "system": system_prompt,
        "name": model_name,
        "details": {
            "parent_model": huggingface_path or "",
            "format": "rkllm",
            "family": family,
            "families": families,
            "parameter_size": parameter_size,
            "quantization_level": quantization_level,
        },
        "model_info": model_info,
        "size": size,
        "capabilities": capabilities,
        "modified_at": modified_at,
    }

    # Add Hugging Face specific fields if available
    if hf_metadata:
        response["huggingface"] = {
            "repo_id": huggingface_path,
            "description": model_description[:500]
            if model_description
            else "",  # Truncate if too long
            "tags": hf_metadata.get("tags", []),
            "downloads": hf_metadata.get("downloads", 0),
            "likes": hf_metadata.get("likes", 0),
        }

    return JSONResponse(response, status_code=200)
