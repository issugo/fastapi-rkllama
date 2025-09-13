import datetime
import os
import shutil

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

import core.config
import core.config.config_utils
import core.model
from api import logger
from api.routes.rkllama import pull_model
from core.model.ModelFile import get_property_modelfile, ModelFile
from core.processing.json_utils import strtobool

router = APIRouter(tags=["ollama"])


@router.post("/api/generate")
async def generate_ollama(request: Request):
    """Generate a response for a given prompt with a provided model. This is a streaming endpoint, so there will be a series of responses. The final response object will include statistics and additional data from the request.
Parameters

    model: (required) the model name
    prompt: the prompt to generate a response for
    suffix: the text after the model response
    images: (optional) a list of base64-encoded images (for multimodal models such as llava)

Advanced parameters (optional):

    format: the format to return a response in. Currently the only accepted value is json
    options: additional model parameters listed in the documentation for the Modelfile such as temperature
    system: system message to (overrides what is defined in the Modelfile)
    template: the prompt template to use (overrides what is defined in the Modelfile)
    context: the context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory
    stream: if false the response will be returned as a single response object, rather than a stream of objects
    raw: if true no formatting will be applied to the prompt. You may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API
    keep_alive: controls how long the model will stay loaded into memory following the request (default: 5m)
"""
    from core.model.ModelFile import ModelFile, get_property_modelfile
    from  main import DEBUG_MODE

    lock_acquired = False  # Track lock status
    is_openai_request = request.path.startswith('/v1/completions')

    try:
        data = request.get_json(force=True)

        if is_openai_request:
            if DEBUG_MODE:
                logger.debug(f"API OpenAI completions request data: {data}")
            data = openai_to_ollama_generate_request(data)

        model_name = core.config.config_utils.get('model')
        prompt = core.config.config_utils.get('prompt')
        system = core.config.config_utils.get('system', '')
        stream = core.config.config_utils.get('stream', True)
        enable_thinking = core.config.config_utils.get('enable_thinking',
                                                       (core.config.config_utils.get('think', None)))  # Ollama now uses 'think' in some versions
        images = core.config.config_utils.get('images', None)  # For multimodal inputs

        # Support format options for structured JSON output
        format_spec = core.config.config_utils.get('format')
        options = core.config.config_utils.get('options', {})

        # Remove possible namespace in model name. Ollama API allows namespace/model
        model_name = re.search(r'/(.*)', model_name).group(1) if re.search(r'/', model_name) else model_name

        if DEBUG_MODE:
            logger.debug(f"API generate request data: {data}")

        if not model_name:
            return JSONResponse({"error": "Missing model name"}, status_code=400)

        if not prompt:
            return JSONResponse({"error": "Missing prompt"}, status_code=400)

        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            model_thinking_enabled = get_property_modelfile(model_name, 'ENABLE_THINKING', core.config.config_utils.get_path("models"))
            enable_thinking = strtobool(model_thinking_enabled) if bool(
                model_thinking_enabled) else False  # Disabled by default

        # Get all model options
        model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="", request_options=options)
        options = model_file.full_options

        # Load model if needed
        if not variables.worker_manager_rkllm.exists_model_loaded(model_name):
            _, error = model_file.load_model()
            if error:
                return JSONResponse(
                    {"error": f"Failed to load model '{model_name}': {error}"},
                    status_code=500,
                )

        # Acquire lock before processing
        variables.verrou.acquire()
        lock_acquired = True

        # DIRECTLY use the GenerateEndpointHandler instead of the process_ollama_generate_request wrapper
        from core.processing.endpoints.GenerateEndpointHandler import GenerateEndpointHandler
        return GenerateEndpointHandler.handle_request(
            model_name=model_name,
            prompt=prompt,
            system=system,
            stream=stream,
            format_spec=format_spec,
            options=options,
            enable_thinking=enable_thinking,
            is_openai_request=is_openai_request,
            images=images
        )
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error in generate_ollama: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            variables.verrou.release()


@router.get("/api/tags")
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


@router.post("/api/pull")
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


@router.delete("/api/delete")
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


@router.post("/api/chat")
@router.post("/v1/chat/completions")
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
        model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="",
                               request_options=options)
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
            model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="",
                                   request_options=options)
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
                model_file = ModelFile(model_name=model_name, endpoint_model_file="", huggingface_path="",
                                       request_options=options)
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


@router.post("/api/embeddings")
def embeddings_ollama():
    # This is a placeholder as embeddings aren't implemented in RKLLAMA
    return JSONResponse(
        {"error": "Embeddings not supported in RKLLAMA"}, status_code=501
    )


@router.get("/api/version")
def ollama_version():
    """Return a dummy version to be compatible with Ollama clients"""
    return JSONResponse({"version": "0.5.1"}, status_code=200)
