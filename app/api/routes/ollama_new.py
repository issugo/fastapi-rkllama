from logging import Logger
from typing import Any, Tuple

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.requests import Request

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
)
from core.model.ModelFile import ModelFileInfo, ModelFile
from core.model.ModelInfo import OllamaModelInfo
from core.model.OllamaManifest import OllamaManifest
from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier
from core.model.storage_helpers.model_pull import Supplier

router = APIRouter(tags=["ollama"])


@router.post("/api/generate", response_model=OllamaGenerateResponse)
async def generate(request: Request, data: OllamaGenerateRequest):
    """
    Generate a response for a given prompt with a provided model.

    This endpoint generates text based on the provided prompt and model.
    If stream is set to true, it will return a streaming response.
    """
    # Default response for demonstration
    if data.stream:
        # Return a streaming response (implementation would depend on your backend)
        async def generate_stream():
            yield OllamaGenerateResponse(
                model=data.model,
                created_at="2025-09-11T12:00:00Z",
                response="This is a streaming response...",
                done=False,
                total_duration=1000000,
                eval_count=10,
            ).model_dump_json().encode() + b"\n"

            yield OllamaGenerateResponse(
                model=data.model,
                created_at="2025-09-11T12:00:01Z",
                response="Generation complete.",
                done=True,
                total_duration=2000000,
                eval_count=20,
            ).model_dump_json().encode() + b"\n"

        return StreamingResponse(generate_stream(), media_type="application/json")

    # Non-streaming response
    return OllamaGenerateResponse(
        model=data.model,
        created_at="2025-09-11T12:00:00Z",
        response="This is a sample generated response based on your prompt: " + data.prompt,
        done=True,
        total_duration=1500000,
        load_duration=200000,
        prompt_eval_duration=300000,
        eval_duration=1000000,
        eval_count=20,
        prompt_eval_count=10,
    )


@router.post("/api/chat", response_model=OllamaChatResponse)
async def chat(request: Request, data: OllamaChatRequest):
    """
    Chat with a model, providing a list of messages.

    This endpoint generates a response based on the conversation history.
    If stream is set to true, it will return a streaming response.
    """
    from core.api.parameters.commons import Role, Message

    # Default response
    if data.stream:
        # Return a streaming response
        async def chat_stream():
            yield OllamaChatResponse(
                model=data.model,
                created_at="2025-09-11T12:00:00Z",
                message=Message(role=Role.ASSISTANT, content="This is a streaming chat response..."),
                done=False,
                total_duration=1000000,
                eval_count=10,
            ).model_dump_json().encode() + b"\n"

            yield OllamaChatResponse(
                model=data.model,
                created_at="2025-09-11T12:00:01Z",
                message=Message(role=Role.ASSISTANT, content="Chat response complete."),
                done=True,
                total_duration=2000000,
                eval_count=20,
            ).model_dump_json().encode() + b"\n"

        return StreamingResponse(chat_stream(), media_type="application/json")

    # Non-streaming response
    return OllamaChatResponse(
        model=data.model,
        created_at="2025-09-11T12:00:00Z",
        message=Message(role=Role.ASSISTANT, content="This is a sample chat response based on your conversation."),
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


@router.get("/api/models", response_model=OllamaListResponse)
async def list_models(request: Request):
    """
    List all available models.

    Returns information about all models that are available locally.
    """
    from core.api.parameters.ollama_commons import OllamaModelInfo

    # Return a sample list of models
    return OllamaListResponse(
        models=[
            OllamaModelInfo(
                name="llama3",
                modified_at="2025-09-01T10:00:00Z",
                size=4_000_000_000,
                digest="sha256:abc123",
                details={"family": "llama", "parameter_size": "8B"},
            ),
            OllamaModelInfo(
                name="mistral",
                modified_at="2025-09-05T14:30:00Z",
                size=5_000_000_000,
                digest="sha256:def456",
                details={"family": "mistral", "parameter_size": "7B"},
            ),
        ]
    )


@router.get("/api/show", response_model=OllamaShowResponse)
async def show_model(request: Request, name: str):
    """
    Show information about a specific model.

    Returns detailed information about a specific model.
    """
    # Return sample model information
    return OllamaShowResponse(
        name=name,
        modified_at="2025-09-01T10:00:00Z",
        size=4_000_000_000,
        digest="sha256:abc123",
        details={
            "family": "llama",
            "parameter_size": "8B",
            "quantization_level": "Q5_K_M",
            "license": "Apache 2.0",
        },
    )


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

        def model_data(self) -> Tuple[str, str, str|None, Supplier]:
            model_name = splitted[0]
            # file contains the model tag when Ollama model
            file = splitted[1]
            # repo is None when Ollama model
            repo = None
            return model_name, file, repo, Supplier.OLLAMA


    if oll_pull_request.stream:
        return pull_model_stream(request=request, pull_supplier=LocalOllamaPullSupplier())
    else:
        error_or_digest = pull_model(request=request, pull_supplier=LocalOllamaPullSupplier())
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
            yield OllamaPushResponse(
                status="uploading model",
                digest="sha256:abc123",
                total=5_000_000_000,
                completed=1_000_000_000,
            ).model_dump_json().encode() + b"\n"

            yield OllamaPushResponse(
                status="uploading model",
                digest="sha256:abc123",
                total=5_000_000_000,
                completed=3_000_000_000,
            ).model_dump_json().encode() + b"\n"

            yield OllamaPushResponse(
                status="verifying upload",
                digest="sha256:abc123",
                total=5_000_000_000,
                completed=5_000_000_000,
            ).model_dump_json().encode() + b"\n"

            yield OllamaPushResponse(
                status="success",
                digest="sha256:abc123",
            ).model_dump_json().encode() + b"\n"

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
            yield OllamaCreateResponse(
                status="processing modelfile",
            ).model_dump_json().encode() + b"\n"

            yield OllamaCreateResponse(
                status="creating model",
            ).model_dump_json().encode() + b"\n"

            yield OllamaCreateResponse(
                status="success",
            ).model_dump_json().encode() + b"\n"

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
