# Import libs
import os
import subprocess
import resource
import argparse
import shutil
import json
import datetime
import re
import sys
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import core.model.ModelFile
from api import api_router
from api.routes.rkllama import pull_model
from core.config import config_utils
from core.config.RKLLAMAConfig import RKLLAMAConfig
from loggers import logging_setup
from core.model.ModelFile import ModelFile, get_property_modelfile

# Local file
from core.processing.json_utils import strtobool
from core.model.ModelPath import find_rkllm_model_name, get_huggingface_model_info


# Check for debug mode using the improved method
DEBUG_MODE = None
logger = None

## app = Flask(__name__)
app = FastAPI()
# Enable CORS for all routes
## CORS(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Original RKLLAMA Routes:
# GET    /models
# POST   /load_model
# POST   /unload_model
# POST   /generate
# POST   /pull
# DELETE /rm




@app.get("/api/tags")
def list_ollama_models():
    # Return models in Ollama API format
    models_dir = core.config.config_utils.get_path("models")

    if not os.path.exists(models_dir):
        return JSONResponse({"models": []}, status_code=200)

    models = []
    for subdir in os.listdir(models_dir):
        subdir_path = os.path.join(models_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".rkllm"):
                    size = os.path.getsize(os.path.join(subdir_path, file))

                    # Extract parameter size and quantization details if available
                    model_details = extract_model_details(file)

                    models.append(
                        {
                            "name": subdir,  # Use simplified name like qwen:3b
                            "model": subdir,  # Match Ollama's format
                            "modified_at": datetime.datetime.fromtimestamp(
                                os.path.getmtime(os.path.join(subdir_path, file))
                            ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                            "size": size,
                            "digest": "",  # Ollama field (not used but included for compatibility)
                            "details": {
                                "format": "rkllm",
                                "family": "llama",  # Default family
                                "parameter_size": core.config.config_utils.get(
                                    "parameter_size", "Unknown"
                                ),
                                "quantization_level": core.config.config_utils.get(
                                    "quantization_level", "Unknown"
                                ),
                            },
                        }
                    )
                    break

    return JSONResponse({"models": models}, status_code=200)


@app.post("/api/show")
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


@app.post("/api/pull")
async def pull_model_ollama(request: Request):
    # TODO: Implement the pull model
    data = await request.json()
    model = core.config.config_utils.get("name")

    if DEBUG_MODE:
        logger.debug(f"API pull request data: {data}")

    if not model:
        return JSONResponse({"error": "Missing model name"}, status_code=400)

    # Ollama API uses application/x-ndjson for streaming
    response_stream = pull_model()  # Call the existing function directly
    response_stream.content_type = "application/x-ndjson"
    return response_stream


@app.delete("/api/delete")
async def delete_model_ollama(request: Request):
    data = await request.json()
    model_name = core.config.config_utils.get("name")

    if DEBUG_MODE:
        logger.debug(f"API delete request data: {data}")

    if not model_name:
        return JSONResponse({"error": "Missing model name"}, status_code=400)

    if not model_name:
        if DEBUG_MODE:
            logger.error(f"Model '{model_name}' not found for deletion")
        return JSONResponse(
            {"error": f"Model '{model_name}' not found"}, status_code=404
        )

    model_path = os.path.join(core.config.config_utils.get_path("models"), model_name)
    if not os.path.exists(model_path):
        return JSONResponse(
            {"error": f"Model directory for '{model_name}' not found"}, status_code=404
        )

    # Check if model is currently loaded
    if current_model == model_name:
        if DEBUG_MODE:
            logger.debug(f"Unloading model '{model_name}' before deletion")
        unload_model()

    try:
        if DEBUG_MODE:
            logger.debug(f"Deleting model directory: {model_path}")
        shutil.rmtree(model_path)

        return JSONResponse({}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to delete model '{model_name}': {str(e)}")
        return JSONResponse(
            {"error": f"Failed to delete model: {str(e)}"}, status_code=500
        )


# Also update the chat endpoint for consistency
@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat_ollama(request: Request):
    lock_acquired = False  # Track lock status
    is_openai_request = request.path.startswith("/v1/chat/completions")

    try:
        data = await request.json()

        if is_openai_request:
            if DEBUG_MODE:
                logger.debug(f"API OpenAI chat request data: {data}")
            data = openai_to_ollama_request(data)

        model_name = core.config.config_utils.get("model")
        messages = core.config.config_utils.get("messages", [])
        system = core.config.config_utils.get("system", "")
        stream = core.config.config_utils.get("stream", True)
        tools = core.config.config_utils.get("tools", None)
        enable_thinking = core.config.config_utils.get("enable_thinking", None)

        # Extract format parameters - can be object or string
        format_spec = core.config.config_utils.get("format")
        options = core.config.config_utils.get("options", {})

        if DEBUG_MODE:
            logger.debug(f"API Ollama chat request data: {data}")

        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            model_thinking_enabled = get_property_modelfile(
                model_name, "ENABLE_THINKING", core.config.config_utils.get_path("models")
            )
            enable_thinking = (
                strtobool(model_thinking_enabled)
                if bool(model_thinking_enabled)
                else False
            )  # Disabled by default

        # Get all model options
        model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="", request_options=options)
        options = model_file.full_options

        # Check if we're starting a new conversation
        # A new conversation is one that doesn't include any assistant messages
        is_new_conversation = not any(
            core.config.config_utils.get("role") == "assistant" for msg in messages
        )

        # Always reset system prompt for new conversations
        if is_new_conversation:
            core.model.ModelFile.system = ""
            if DEBUG_MODE:
                logger.debug("New conversation detected, resetting system prompt")

        # Extract system message from messages array if present
        system_in_messages = False
        filtered_messages = []

        for message in messages:
            if core.config.config_utils.get("role") == "system":
                system = core.config.config_utils.get("content", "")
                system_in_messages = True
                # Don't add system message to filtered messages
            else:
                filtered_messages.append(message)

        # Only use the extracted system message or explicit system parameter if provided
        if system_in_messages or system:
            core.model.ModelFile.system = system
            messages = filtered_messages
            if DEBUG_MODE:
                logger.debug(f"Using system message: {system}")

        # Load model if needed
        if core.endpoints.GlobalState.current_model != model_name:
            if core.endpoints.GlobalState.current_model:
                if DEBUG_MODE:
                    logger.debug(f"Unloading current model: {core.endpoints.GlobalState.current_model}")
                unload_model()

            if DEBUG_MODE:
                logger.debug(f"Loading model: {model_name}")
            model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="", request_options=options)
            modele_instance, error = model_file.load_model()
            if error:
                if DEBUG_MODE:
                    logger.error(f"Failed to load model {model_name}: {error}")
                return JSONResponse(
                    {"error": f"Failed to load model '{model_name}': {error}"},
                    status_code=500,
                )
            core.endpoints.GlobalState.rkllm_model = modele_instance
            core.endpoints.GlobalState.current_model = model_name
            if DEBUG_MODE:
                logger.debug(f"Model {model_name} loaded successfully")
        else:
            # If model is already loaded, check its options are the same for the current request
            if (
                core.endpoints.GlobalState.rkllm_model.rkllm_param.max_context_len
                != int(core.config.config_utils.get("num_ctx"))
                or core.endpoints.GlobalState.rkllm_model.rkllm_param.max_new_tokens
                != int(core.config.config_utils.get("max_new_tokens"))
                or core.endpoints.GlobalState.rkllm_model.rkllm_param.top_k != int(
                core.config.config_utils.get("top_k"))
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.top_p, 2)
                != round(float(core.config.config_utils.get("top_p")), 2)
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.temperature, 2)
                != round(float(core.config.config_utils.get("temperature")), 2)
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.repeat_penalty, 2)
                != round(float(core.config.config_utils.get("repeat_penalty")), 2)
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.frequency_penalty, 2)
                != round(float(core.config.config_utils.get("frequency_penalty")), 2)
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.presence_penalty, 2)
                != round(float(core.config.config_utils.get("presence_penalty")), 2)
                or core.endpoints.GlobalState.rkllm_model.rkllm_param.mirostat
                != int(core.config.config_utils.get("mirostat"))
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.mirostat_tau, 2)
                != round(float(core.config.config_utils.get("mirostat_tau")), 2)
                or round(core.endpoints.GlobalState.rkllm_model.rkllm_param.mirostat_eta, 2)
                != round(float(core.config.config_utils.get("mirostat_eta")), 2)
            ):
                # Update model parameters if they differ
                if DEBUG_MODE:
                    logger.debug(
                        f"Updating model parameters for {model_name} with options: {options}"
                    )

                if core.endpoints.GlobalState.current_model:
                    if DEBUG_MODE:
                        logger.debug(f"Unloading current model: {core.endpoints.GlobalState.current_model}")
                    unload_model()

                if DEBUG_MODE:
                    logger.debug(f"Reoading model: {model_name}")
                model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="", request_options=options)
                modele_instance, error = model_file.load_model()
                if error:
                    if DEBUG_MODE:
                        logger.error(f"Failed to reload model {model_name}: {error}")
                    return JSONResponse(
                        {"error": f"Failed to reload model '{model_name}': {error}"},
                        status_code=500,
                    )
                core.endpoints.GlobalState.rkllm_model = modele_instance
                core.endpoints.GlobalState.current_model = model_name
                if DEBUG_MODE:
                    logger.debug(f"Model {model_name} reloaded successfully")

        # Store format settings in model instance
        if core.endpoints.GlobalState.rkllm_model:
            core.endpoints.GlobalState.rkllm_model.format_schema = format_spec
            core.endpoints.GlobalState.rkllm_model.format_options = options

        # Acquire lock before processing the request
        variables.verrou.acquire()
        lock_acquired = True  # Mark lock as acquired

        # Create custom request for processing
        custom_req = type(
            "obj",
            (object,),
            {
                "json": {
                    "model": model_name,
                    "messages": messages,
                    "stream": stream,
                    "system": system,
                    "format": format_spec,
                    "options": options,
                    "tools": tools,
                    "enable_thinking": enable_thinking,
                },
                "path": "/api/chat",
            },
        )

        # Set a flag on the custom request to indicate it should not release the lock
        # as we'll handle it here
        custom_req.handle_lock = False

        # Process the request - this won't release the lock
        from core.processing.endpoints.ChatEndpointHandler import ChatEndpointHandler

        return ChatEndpointHandler.handle_request(
            modele_rkllm=core.endpoints.GlobalState.rkllm_model,
            model_name=model_name,
            messages=messages,
            system=system,
            stream=stream,
            format_spec=format_spec,
            options=options,
            tools=tools,
            enable_thinking=enable_thinking,
            is_openai_request=is_openai_request,
        )

    except Exception as e:
        logger.exception("Error in chat_ollama")
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            if DEBUG_MODE:
                logger.debug("Releasing lock in chat_ollama")
            variables.verrou.release()


# Only include debug endpoint if in debug mode
if DEBUG_MODE:

    @app.post("/api/debug")
    async def debug_streaming(request: Request):
        """Endpoint to diagnose streaming issues"""
        data = await request.json()
        stream_data = core.config.config_utils.get("stream_data", "")

        issues = check_response_format(stream_data)

        if issues:
            return JSONResponse(
                {
                    "status": "error",
                    "issues": issues,
                    "recommendation": "Check server_utils.py implementation of streaming",
                },
                status_code=200,
            )
        else:
            return JSONResponse(
                {"status": "ok", "message": "No issues found in the response format"},
                status_code=200,
            )


@app.post("/api/embeddings")
def embeddings_ollama():
    # This is a placeholder as embeddings aren't implemented in RKLLAMA
    return JSONResponse(
        {"error": "Embeddings not supported in RKLLAMA"}, status_code=501
    )


# Version endpoint for Ollama API compatibility
@app.get("/api/version")
def ollama_version():
    """Return a dummy version to be compatible with Ollama clients"""
    return JSONResponse({"version": "0.5.1"}, status_code=200)


# Default route


# Launch function
def main():
    # Define the arguments for the launch function
    parser = argparse.ArgumentParser(
        description="RKLLM server initialization with configurable options."
    )
    parser.add_argument("--processor", type=str, help="Processor: rk3588/rk3576.")
    parser.add_argument("--port", type=str, help="Port for the server")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()


    config_utils.rkllama_config = RKLLAMAConfig(app_root=os.getenv("APP_ROOT"), args=args)

    # Set debug mode if specified in config - using the improved method
    global DEBUG_MODE
    DEBUG_MODE = config_utils.rkllama_config.is_debug_mode()

    logging_setup(config_utils.rkllama_config.get_path("logs"), DEBUG_MODE)

    global logger
    logger = logging.getLogger("rkllama.server")

    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.warning("Debug mode enabled")
        config_utils.rkllama_config.display()
        os.environ["RKLLAMA_DEBUG"] = "1"  # Explicitly set for subprocess consistency

    # Get port from config
    port = config_utils.rkllama_config.server.port

    # Check the processor
    processor = config_utils.rkllama_config.platform.processor
    if not processor:
        logger.error("Processor not configured")
        sys.exit(1)
    else:
        if processor not in ["rk3588", "rk3576"]:
            logger.error(
                "Error: Invalid processor. Please enter rk3588 or rk3576."
            )
            sys.exit(1)
        logger.info(f"Setting the frequency for the {processor} platform...")
        library_path = os.path.join(config_utils.rkllama_config.get_path("lib"), f"fix_freq_{processor}.sh")

        # Pass debug flag as parameter to the shell script
        debug_param = "1" if DEBUG_MODE else "0"
        command = f"sudo bash {library_path} {debug_param}"
        subprocess.run(command, shell=True)

    # Set the resource limits
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Start the API server with the chosen port
    logger.info(f"Starting the API at http://localhost:{port}")

    # Set Flask debug mode to match our debug flag
    ## flask_debug = config.is_debug_mode()
    ## app.run(host=config.get("server", "host", "0.0.0.0"), port=int(port), threaded=True, debug=flask_debug)
    uvicorn.run(
        app,
        host=config_utils.rkllama_config.server.host,
        port=int(port),
        log_level="debug",
    )


if __name__ == "__main__":
    main()
