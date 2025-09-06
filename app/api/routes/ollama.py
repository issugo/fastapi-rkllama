from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

from core import config, model
from core.model.Model import unload_model
from core.model.ModelFile import load_model
from main import app, DEBUG_MODE, logger
from src import variables as variables
from src.format_utils import strtobool
from src.model_utils import get_property_modelfile, get_model_full_options

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

    lock_acquired = False  # Track lock status
    is_openai_request = request.path.startswith('/v1/completions')

    try:
        data = request.get_json(force=True)

        if is_openai_request:
            if DEBUG_MODE:
                logger.debug(f"API OpenAI completions request data: {data}")
            data = openai_to_ollama_generate_request(data)

        model_name = data.get('model')
        prompt = data.get('prompt')
        system = data.get('system', '')
        stream = data.get('stream', True)
        enable_thinking = data.get('enable_thinking',
                                   (data.get('think', None)))  # Ollama now uses 'think' in some versions
        images = data.get('images', None)  # For multimodal inputs

        # Support format options for structured JSON output
        format_spec = data.get('format')
        options = data.get('options', {})

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
            model_thinking_enabled = get_property_modelfile(model_name, 'ENABLE_THINKING', config.get_path("models"))
            enable_thinking = strtobool(model_thinking_enabled) if bool(
                model_thinking_enabled) else False  # Disabled by default

        # Get all model options
        options = get_model_full_options(model_name, config.get_path("models"), options)

        # Load model if needed
        if not variables.worker_manager_rkllm.exists_model_loaded(model_name):
            _, error = load_model(model_name, request_options=options)
            if error:
                return JSONResponse(
                    {"error": f"Failed to load model '{model_name}': {error}"},
                    status_code=500,
                )

        # Acquire lock before processing
        variables.verrou.acquire()
        lock_acquired = True

        # DIRECTLY use the GenerateEndpointHandler instead of the process_ollama_generate_request wrapper
        from src.server_utils import GenerateEndpointHandler
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

    lock_acquired = False  # Track lock status

    try:
        data = await request.json()
        model_name = config.get("model")
        prompt = config.get("prompt")
        system = config.get("system", "")
        stream = config.get("stream", True)
        enable_thinking = config.get("enable_thinking", None)

        # Support format options for structured JSON output
        format_spec = config.get("format")
        options = config.get("options", {})

        if DEBUG_MODE:
            logger.debug(f"API generate request data: {data}")

        if not model_name:
            return JSONResponse({"error": "Missing model name"}, status_code=400)

        if not prompt:
            return JSONResponse({"error": "Missing prompt"}, status_code=400)

        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            model_thinking_enabled = get_property_modelfile(
                model_name, "ENABLE_THINKING", config.get_path("models")
            )
            enable_thinking = (
                strtobool(model_thinking_enabled)
                if bool(model_thinking_enabled)
                else False
            )  # Disabled by default

        # Get all model options
        options = get_model_full_options(model_name, config.get_path("models"), options)

        # Load model if needed
        if model.current_model != model_name:
            if model.current_model:
                unload_model()
            modele_instance, error = load_model(model_name, request_options=options)
            if error:
                return JSONResponse(
                    {"error": f"Failed to load model '{model_name}': {error}"},
                    status_code=500,
                )
            model.modele_rkllm = modele_instance
            model.current_model = model_name

        # Acquire lock before processing
        variables.verrou.acquire()
        lock_acquired = True

        # DIRECTLY use the GenerateEndpointHandler instead of the process_ollama_generate_request wrapper
        from src.server_utils import GenerateEndpointHandler

        return GenerateEndpointHandler.handle_request(
            modele_rkllm=model.modele_rkllm,
            model_name=model_name,
            prompt=prompt,
            system=system,
            stream=stream,
            format_spec=format_spec,
            options=options,
            enable_thinking=enable_thinking,
        )
    except Exception as e:
        if DEBUG_MODE:
            logger.exception(f"Error in generate_ollama: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Only release if we acquired it
        if lock_acquired and variables.verrou.locked():
            variables.verrou.release()
