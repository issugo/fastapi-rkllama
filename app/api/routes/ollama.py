import datetime
from logging import Logger
from typing import Any, Tuple, List

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.requests import Request
from starlette.responses import JSONResponse

from api import logger
from core.api.parameters.ollama_requests import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OllamaEmbeddingRequest,
    OllamaPullRequest,
    OllamaPushRequest,
    OllamaCreateRequest,
    OllamaCopyRequest,
    OllamaDeleteRequest,
    OllamaShowRequest,
)
from core.api.parameters.ollama_responses import (
    OllamaGenerateResponse,
    OllamaChatResponse,
    OllamaEmbeddingResponse,
    OllamaListResponse,
    OllamaShowResponse,
    OllamaPullResponse,
    OllamaPushResponse,
    OllamaCreateResponse,
    OllamaCopyResponse,
    OllamaDeleteResponse,
    OllamaHFModelShow,
    OllamaModelShowDetails,
    OllamaPsResponse,
    OllamaProcessModel,
)
from core.backends.backend import BackendType
from core.config.RKLLAMAConfig import RKLLAMASettings
from core.model.Model import Model
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.model.ModelPath import ModelDirException, ModelPath, ModelException
from core.model.ModelType import FILE_TYPE, ModelType
from core.model.models_constants import (
    LANGUAGE_DEFAULT,
    default_context_length,
    DEFAULT_SYSTEM,
    DEFAULT_TEMPLATE,
)
from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier
from core.model.storage_helpers.SupplierFileInfo import Supplier
from core.processing.WorkerManager import WorkerManager, get_worker_manager
from core.processing.api_handlers.ollama_api_handler import (
    OllamaGenerateAPIHandler,
    OllamaChatAPIHandler,
)
from core.processing.endpoints.ChatEndpointHandler import ChatEndpointHandler
from core.processing.endpoints.GenerateEndpointHandler import GenerateEndpointHandler

router = APIRouter(tags=["ollama"])

setting: RKLLAMASettings | None = None
settings: RKLLAMASettings | None = None
DEBUG_MODE: bool | None = None


@router.post("/api/pull", response_model=OllamaPullResponse)
async def pull_model(request: Request, oll_pull_request: OllamaPullRequest):
    """
    Pull a model from a registry.

    Downloads a model from the Ollama library or a specified registry.
    If stream parameter is set to true, it will return a streaming response with progress updates.
    """

    from core.model.storage_helpers.model_pull import pull_model, pull_model_stream

    splitted = oll_pull_request.name.split(":")

    class LocalOllamaPullSupplier(OllamaPullSupplier):
        @property
        def logger(self) -> Logger:
            return logger

        def check_params(self) -> Any | None:
            if len(splitted) < 2:
                return self.error(f"Invalid path '{oll_pull_request.model}'")
            return None

        def model_data(self) -> Tuple[str, str, str | None, Supplier]:
            model_name = splitted[0]
            # file contains the model tag when Ollama model
            file = splitted[1]
            # repo is None when Ollama model
            repo = None
            return model_name, file, repo, Supplier.OLLAMA

    if oll_pull_request.stream:
        return pull_model_stream(
            request=request, pull_supplier=LocalOllamaPullSupplier()
        )
    else:
        error_or_digest = pull_model(
            request=request, pull_supplier=LocalOllamaPullSupplier()
        )
        # Non-streaming response
        if error_or_digest:
            if error_or_digest.startswith("Error:"):
                return OllamaPullResponse(
                    status=f"{error_or_digest}",
                )
            return OllamaPullResponse(
                status="success",
                digest=f"sha256:{error_or_digest}",
            )
        return OllamaPullResponse(
            status="success",
        )


@router.get("/api/tags", response_model=OllamaListResponse)
async def list_models(request: Request):
    """
    List all available models.

    Returns information about all models that are available locally.
    """
    from core.api.parameters.ollama_commons import OllamaModelInfo

    try:
        model_list: List[Model] = Model.list()

        # Return a sample list of models
        return OllamaListResponse(
            models=[OllamaModelInfo.from_model(model) for model in model_list]
        )
    except ModelDirException as mde:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(mde)}."}),
            status_code=500,
        )


@router.get("/api/ps", response_model=OllamaPsResponse)
async def list_loaded_models(request: Request):
    """
    List loaded models.

    Returns information about currently loaded models in memory.
    """
    from core.processing.WorkerManager import worker_managers
    from core.api.parameters.ollama_commons import OllamaModelInfoDetails

    models_running = []
    try:
        model_list: List[Model] = Model.list()
        models_info = {m.id: m for m in model_list}

        for wm in worker_managers:
            for model_id, worker in wm.workers.items():
                worker_model_info = worker.worker_model_info

                info = models_info.get(model_id)
                digest = info.digest if info else ""
                format_val = info.model_info.details.model_format if info else "rkllm"
                family_val = info.model_info.details.model_family if info else "qwen2"
                parameter_size_val = (
                    info.model_info.details.parameter_size if info else "unknown"
                )
                quantization_level_val = (
                    info.model_info.details.quantization_level if info else "unknown"
                )

                models_running.append(
                    OllamaProcessModel(
                        name=model_id,
                        model=model_id,
                        modified_at=(
                            info.model_info.modified_at_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                            if info
                            else datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        ),
                        size=worker_model_info.size,
                        digest=digest,
                        details=OllamaModelInfoDetails(
                            format=format_val,
                            family=family_val,
                            parameter_size=parameter_size_val,
                            quantization_level=quantization_level_val,
                        ),
                        expires_at=worker_model_info.expires_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        size_vram=worker_model_info.size,
                    )
                )
    except Exception as e:
        logger.exception("Error listing running models")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing running models: {str(e)}"
        )

    return OllamaPsResponse(models=models_running)


@router.get("/api/show/{model_id}", response_model=OllamaShowResponse)
async def show_model_by_id(request: Request, model_id: str):
    return await show_model(request, OllamaShowRequest(name=model_id))


@router.post("/api/show", response_model=OllamaShowResponse)
async def show_model(request: Request, ollama_show_request: OllamaShowRequest):
    """
    Show information about a specific model.

    Returns detailed information about a specific model.
    """
    try:
        name: str = ollama_show_request.name
        model_path: ModelPath = ModelPath.from_model_id(model_id=name)
        model: Model = Model.load(model_path=model_path)

        parameters_str: str = model.model_info.details.parameter_size
        parameter_size: int = model.model_metadata.parameters
        license_text = None
        modelfile_content = None

        # Build a more complete model_info dict with architecture details
        model_info = {
            "general.architecture": model.model_metadata.architecture,
            "general.base_model.0.name": f"{model.model_info.name} {model.model_info.details.parameter_size}",
            "general.base_model.0.organization": model.model_info.details.model_family.capitalize(),
            "general.basename": model.model_info.name,
            "general.file_type": FILE_TYPE.get(
                model.model_info.model_type
            ),  # RKLLM file type
            "general.parameter_count": parameter_size,
            "general.quantization_version": 2,
            "general.size_label": parameters_str,
            "general.tags": model.model_info.tags,
            "general.type": "model",
            "tokenizer.ggml.pre": model.model_metadata.architecture,
        }

        # Add repo URL if available
        repo_url = model.model_path.repo_url
        if repo_url:
            model_info["general.base_model.0.repo_url"] = repo_url
            model_info["general.base_model.count"] = 1

        _model_description = (
            model.model_metadata.description
        )  # noqa: F841 to be change if Modelfile

        # Add finetune info if available
        if model.model_metadata.finetune:
            model_info["general.finetune"] = model.model_metadata.finetune

        # Add license info if available
        if model.model_metadata.license:
            license_text = model.model_metadata.license.license_text
            license_name = model.model_metadata.license.common_license
            license_link = model.model_metadata.license.license_link
            model_info["general.license"] = license_name
            if license_link:
                model_info["general.license.link"] = license_link
            model_info["general.license.name"] = license_name

        # Add language info if we can detect it
        if model.model_info.languages:
            model_info["general.languages"] = model.model_info.languages
        else:
            # Default to English
            model_info["general.languages"] = [LANGUAGE_DEFAULT]

        # Add system info if available
        system_prompt = model.model_metadata.system_prompt or DEFAULT_SYSTEM
        # Add system info if available
        template = model.model_metadata.template or DEFAULT_TEMPLATE

        # load Modelfile if exists,
        if model_path.modelfile_exists:
            if model_path.modelfile_match:
                logger.info(f"Modelfile found for {name}")
                modelfile: ModelFile = ModelFile.load(model_path=model_path)
                modelfile_content = modelfile.content()
                #  then update license if Modelfile contains license data,
                if modelfile.LICENSE:
                    license_text = modelfile.license.license_text
                    license_name = modelfile.license.common_license
                    license_link = modelfile.license.license_link
                    model_info["general.license"] = license_name
                    if license_link:
                        model_info["general.license.link"] = license_link
                    model_info["general.license.name"] = license_name
                #  then update system if Modelfile contains system data,
                if modelfile.SYSTEM:
                    system_prompt = modelfile.SYSTEM
                #  then update template if Modelfile contains template data,
                if modelfile.TEMPLATE:
                    template = modelfile.TEMPLATE

        # Add architecture-specific parameters based on model family
        family = model.model_metadata.architecture
        if family == "qwen2":
            model_info.update(
                {
                    "qwen2.attention.head_count": 16,
                    "qwen2.attention.head_count_kv": 2,
                    "qwen2.attention.layer_norm_rms_epsilon": 0.000001,
                    "qwen2.block_count": 36 if "3B" in parameters_str else 24,
                    "qwen2.context_length": default_context_length(family),
                    "qwen2.embedding_length": 2048 if "3B" in parameters_str else 1536,
                    "qwen2.feed_forward_length": (
                        11008 if "3B" in parameters_str else 8192
                    ),
                    "qwen2.rope.freq_base": 1000000,
                }
            )
        elif family in ["llama", "llama2", "llama3"]:
            model_info.update(
                {
                    f"{family}.attention.head_count": 32,
                    f"{family}.attention.head_count_kv": 4,
                    f"{family}.attention.layer_norm_rms_epsilon": 0.000001,
                    f"{family}.block_count": 32,
                    f"{family}.context_length": default_context_length(family),
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
                    "mistral.context_length": default_context_length(family),
                    "mistral.embedding_length": 4096,
                    "mistral.feed_forward_length": 14336,
                }
            )

        # Prepare response with enhanced metadata
        # Add Hugging Face specific fields if available
        ollama_hf_model_show = None
        if model.model_path.huggingface_model_info_exists:
            ollama_hf_model_show: OllamaHFModelShow = OllamaHFModelShow(
                repo_id=model.model_info.hf_model_info.id,
                description=(
                    model.model_info.hf_model_info.description[:500]
                    if model.model_info.hf_model_info.description
                    else ""
                ),
                tags=model.model_info.hf_model_info.tags,
                downloads=model.model_info.hf_model_info.downloads,
                likes=model.model_info.hf_model_info.likes,
            )

        # TODO: use OllamaShowResponse
        return OllamaShowResponse(
            license=license_text or "Unknown",
            modelfile=modelfile_content or "",  # file content
            parameters=parameters_str,
            template=template,
            system=system_prompt,
            name=model.model_info.name,
            details=OllamaModelShowDetails(
                parent_model=model.model_info.base_model or "",
                format=model.model_metadata.model_type.name.lower(),
                family=family,
                families=model.model_info.details.model_families,
                parameter_size=parameters_str,
                quantization_level=model.model_info.details.quantization_level,
            ),
            model_info=model_info,
            size=model.size,
            digest=f"sha256:{model.digest}",
            capabilities=model.model_info.capabilities,
            modified_at=model.model_info.modified_at_dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            huggingface=ollama_hf_model_show,
        )
    except ModelException as me:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(me)}."}),
            status_code=500,
        )


@router.post("/api/generate", response_model=OllamaGenerateResponse)
async def generate(request: Request, ollama_generate_request: OllamaGenerateRequest):
    """
    Generate a response for a given prompt with a provided model.

    This endpoint generates text based on the provided prompt and model.
    If stream is set to true, it will return a streaming response.
    """

    global settings
    if settings is None:
        from core.config import config_utils

        settings = config_utils.get_settings()

    global DEBUG_MODE
    if DEBUG_MODE is None:
        DEBUG_MODE = settings.is_debug_mode()

    if DEBUG_MODE:
        logger.debug(
            f"API generate request OllamaGenerateRequest: {ollama_generate_request}"
        )

    if not ollama_generate_request.prompt:
        return JSONResponse(
            jsonable_encoder({"error": "Missing prompt"}),
            status_code=400,
        )

    try:
        # modelfile can be search
        model_path: ModelPath = ModelPath.from_model_id(
            model_id=ollama_generate_request.model
        )
        modelfile: ModelFile = ModelFile.load(model_path=model_path)

        # optional non formatted values
        try:
            data = await request.json()
        except Exception:
            data = {}
        # enable_thinking is None if not specified
        enable_thinking = data.get(
            "enable_thinking", (data.get("think", None))
        )  # Ollama now uses 'think' in some versions
        _images = data.get("images", None)  # noqa: F841 For multimodal inputs

        full_model_parameters: FullModelParameters = modelfile.full_model_parameters
        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            enable_thinking = (
                full_model_parameters.enable_thinking
                if (full_model_parameters.enable_thinking is not None)
                else False
            )  # Disabled by default

        full_model_parameters: FullModelParameters = (
            FullModelParameters.ollama_override(
                full_model_parameters=full_model_parameters,
                ollama_options=ollama_generate_request.options,
                enable_thinking=enable_thinking,
            )
        )

        model: Model = modelfile.model
        model_type: ModelType = model.model_type
        worker_manager: WorkerManager = get_worker_manager(
            backend_type=BackendType.from_model_type(model_type=model_type)
        )

        # Model loaded into memory
        model_worker, model_process = worker_manager.add_worker(
            modelfile=modelfile, full_model_parameters=full_model_parameters
        )

        return GenerateEndpointHandler.handle_request(
            model_worker=model_worker,
            api_handler=OllamaGenerateAPIHandler(),
            modelfile=modelfile,
            prompt=ollama_generate_request.prompt,
            system=ollama_generate_request.system,
            stream=ollama_generate_request.stream,
            options=full_model_parameters,
            enable_thinking=enable_thinking,
            format_spec=ollama_generate_request.format,
        )
    except ModelException as me:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(me)}."}),
            status_code=500,
        )


@router.post("/api/chat", response_model=OllamaChatResponse)
async def chat(request: Request, ollama_chat_request: OllamaChatRequest):
    """
    Chat with a model, providing a list of messages.

    This endpoint generates a response based on the conversation history.
    If stream is set to true, it will return a streaming response.
    """
    from core.api.parameters.commons import Role, Message

    # lock_acquired = False  # Track lock status

    global settings
    if settings is None:
        from core.config import config_utils

        settings = config_utils.get_settings()

    global DEBUG_MODE
    if DEBUG_MODE is None:
        DEBUG_MODE = settings.is_debug_mode()

    if DEBUG_MODE:
        logger.debug(f"API chat request OllamaChatRequest: {ollama_chat_request}")

    if not ollama_chat_request.messages:
        return JSONResponse(
            jsonable_encoder({"error": "Missing messages"}),
            status_code=400,
        )

    messages: List[Message] = ollama_chat_request.messages
    try:
        # modelfile can be search
        model_path: ModelPath = ModelPath.from_model_id(
            model_id=ollama_chat_request.model
        )
        modelfile: ModelFile = ModelFile.load(model_path=model_path)

        # optional non formatted values
        try:
            data = await request.json()
        except Exception:
            data = {}
        # enable_thinking is None if not specified
        enable_thinking = data.get(
            "enable_thinking", (data.get("think", None))
        )  # Ollama now uses 'think' in some versions
        # Review the images in messages
        images = data.get("images", None)  # For multimodal inputs
        tools = data.get("tools", None)

        full_model_parameters: FullModelParameters = modelfile.full_model_parameters
        # Get Thinking setting from modelfile if not provided
        if enable_thinking is None:
            enable_thinking = (
                full_model_parameters.enable_thinking
                if (full_model_parameters.enable_thinking is not None)
                else False
            )  # Disabled by default

        full_model_parameters: FullModelParameters = (
            FullModelParameters.ollama_override(
                full_model_parameters=full_model_parameters,
                ollama_options=ollama_chat_request.options,
                enable_thinking=enable_thinking,
            )
        )

        model: Model = modelfile.model
        model_type: ModelType = model.model_type
        worker_manager: WorkerManager = get_worker_manager(
            backend_type=BackendType.from_model_type(model_type=model_type)
        )

        # Check if we're starting a new conversation
        # A new conversation is one that doesn't include any assistant messages
        is_new_conversation = not any(
            msg.role == "assistant" for msg in ollama_chat_request.messages
        )

        # Always reset system prompt for new conversations
        if is_new_conversation:
            system = (
                modelfile.SYSTEM
                if modelfile.SYSTEM
                else model.model_metadata.system_prompt
            )
            if DEBUG_MODE:
                logger.debug("New conversation detected, resetting system prompt")

        # Extract system message from messages array if present
        system_in_messages = False
        msg_system = ""
        filtered_messages = []

        for message in ollama_chat_request.messages:
            if message.role == "system":
                msg_system = message.content if message.content else ""
                system_in_messages = True
                # Don't add system message to filtered messages
            else:
                filtered_messages.append(message)
                # CHeck for images in user messages for multimodal
                if message.role == "user" and "images" in message:
                    if "images" not in data:
                        data["images"] = []
                    data["images"].extend(message["images"])

        # Only use the extracted system message or explicit system parameter if provided
        if system_in_messages or msg_system:
            system = msg_system
            messages = filtered_messages
            if DEBUG_MODE:
                logger.debug(f"Using system message: {system}")

        if system is None:
            system = (
                modelfile.SYSTEM
                if modelfile.SYSTEM
                else model.model_metadata.system_prompt
            )

        # Model loaded into memory
        model_worker, model_process = worker_manager.add_worker(
            modelfile=modelfile, full_model_parameters=full_model_parameters
        )

        # Store format settings in model instance
        # if rkllm_model_request:
        #    rkllm_model_request.format_schema = format_spec
        #    rkllm_model_request.format_options = options

        # Process the request - this won't release the lock
        return ChatEndpointHandler.handle_request(
            model_worker=model_worker,
            api_handler=OllamaChatAPIHandler(),
            modelfile=modelfile,
            messages=messages,
            system=system,
            stream=ollama_chat_request.stream,
            tools=tools,
            images=images,
            options=full_model_parameters,
            enable_thinking=enable_thinking,
            format_spec=ollama_chat_request.format,
        )

    except ModelException as me:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(me)}."}),
            status_code=500,
        )

    # Default response
    if data.stream:
        # Return a streaming response
        async def chat_stream():
            yield (
                OllamaChatResponse(
                    model=data.model,
                    created_at="2025-09-11T12:00:00Z",
                    message=Message(
                        role=Role.ASSISTANT,
                        content="This is a streaming chat response...",
                    ),
                    done=False,
                    total_duration=1000000,
                    eval_count=10,
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

            yield (
                OllamaChatResponse(
                    model=data.model,
                    created_at="2025-09-11T12:00:01Z",
                    message=Message(
                        role=Role.ASSISTANT, content="Chat response complete."
                    ),
                    done=True,
                    total_duration=2000000,
                    eval_count=20,
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

        return StreamingResponse(chat_stream(), media_type="application/json")

    # Non-streaming response
    return OllamaChatResponse(
        model=data.model,
        created_at="2025-09-11T12:00:00Z",
        message=Message(
            role=Role.ASSISTANT,
            content="This is a sample chat response based on your conversation.",
        ),
        done=True,
        total_duration=1500000,
        load_duration=200000,
        prompt_eval_duration=300000,
        eval_count=20,
        prompt_eval_count=10,
    )


@router.post("/api/embeddings", response_model=OllamaEmbeddingResponse)
async def embeddings(request: Request, data: OllamaEmbeddingRequest):
    """
    Generate embeddings for the given prompt.

    This endpoint generates vector embeddings that can be used for semantic search, clustering, etc.
    """
    # Return a sample embedding (normally would be much longer)
    return OllamaEmbeddingResponse(
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]
    )


@router.post("/api/push", response_model=OllamaPushResponse)
async def push_model(request: Request, data: OllamaPushRequest):
    """
    Push a model to a registry.

    Uploads a model to the Ollama library or a specified registry.
    If stream is set to true, it will return a streaming response with progress updates.
    """
    if data.stream:
        # Return a streaming response with progress updates
        async def push_stream():
            yield (
                OllamaPushResponse(
                    status="uploading model",
                    digest="sha256:abc123",
                    total=5_000_000_000,
                    completed=1_000_000_000,
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

            yield (
                OllamaPushResponse(
                    status="uploading model",
                    digest="sha256:abc123",
                    total=5_000_000_000,
                    completed=3_000_000_000,
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

            yield (
                OllamaPushResponse(
                    status="verifying upload",
                    digest="sha256:abc123",
                    total=5_000_000_000,
                    completed=5_000_000_000,
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

            yield (
                OllamaPushResponse(
                    status="success",
                    digest="sha256:abc123",
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

        return StreamingResponse(push_stream(), media_type="application/json")

    # Non-streaming response
    return OllamaPushResponse(
        status="success",
        digest="sha256:abc123",
    )


@router.post("/api/create", response_model=OllamaCreateResponse)
async def create_model(request: Request, data: OllamaCreateRequest):
    """
    Create a model.

    Creates a new model from a Modelfile.
    If stream is set to true, it will return a streaming response with progress updates.
    """
    if data.stream:
        # Return a streaming response with progress updates
        async def create_stream():
            yield (
                OllamaCreateResponse(
                    status="processing modelfile",
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

            yield (
                OllamaCreateResponse(
                    status="creating model",
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

            yield (
                OllamaCreateResponse(
                    status="success",
                )
                .model_dump_json()
                .encode()
                + b"\n"
            )

        return StreamingResponse(create_stream(), media_type="application/json")

    # Non-streaming response
    return OllamaCreateResponse(
        status="success",
    )


@router.post("/api/copy", response_model=OllamaCopyResponse)
async def copy_model(request: Request, data: OllamaCopyRequest):
    """
    Copy a model.

    Creates a copy of a model with a new name.
    """
    return OllamaCopyResponse(
        status=f"copied model from {data.source} to {data.destination}",
    )


@router.delete("/api/delete", response_model=OllamaDeleteResponse)
async def delete_model(request: Request, data: OllamaDeleteRequest):
    """
    Delete a model.

    Removes a model from local storage.
    """
    return OllamaDeleteResponse(
        status=f"deleted model {data.name}",
    )
