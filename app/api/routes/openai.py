import asyncio
import json
import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from core.api.parameters.commons import Usage
from core.api.parameters.openai_commons import (
    OpenAIFinishReason,
    OpenAIImageDetail,
    OpenAIModel,
)
from core.api.parameters.openai_requests import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ModerationRequest,
    ImageGenerationRequest,
    ImageEditRequest,
    ImageVariationRequest,
    VisionRequest,
)
from core.api.parameters.openai_responses import (
    ChatCompletionResponse,
    ChatCompletionChunkResponse,
    CompletionResponse,
    EmbeddingResponse,
    ModerationResponse,
    ImageGenerationResponse,
    ImageEditResponse,
    ImageVariationResponse,
    VisionResponse,
    ChatCompletionChoice,
    CompletionChoice,
    ModerationResult,
    ImageData,
)
from core.model.Model import Model
from core.model.ModelPath import ModelDirException, ModelPath, ModelException

# Create router with "openai" tag
router = APIRouter(tags=["openai"])


@router.get("/v1/models", response_model=List[OpenAIModel])
async def list_models(request: Request):
    """
    List available models that can be used with the API.

    The model object has OpenAIModel type.
    """

    try:
        model_list: List[Model] = Model.list()

        # Return a sample list of models
        return [OpenAIModel.from_model(model) for model in model_list]
    except ModelDirException as mde:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(mde)}."}),
            status_code=500,
        )


@router.get("/v1/models/{model_id}", response_model=OpenAIModel)
def get_model(model_id: str):
    try:
        model: Model = Model.load(model_path=ModelPath.from_model_id(model_id))
        return OpenAIModel.from_model(model)
    except ModelException as me:
        return JSONResponse(
            jsonable_encoder({"error": f"{str(me)}."}),
            status_code=500,
        )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: Request, chat_request: ChatCompletionRequest):
    """
    Create a completion for the chat message.
    """
    try:
        model = chat_request.model
        messages = chat_request.messages
        stream = chat_request.stream or False

        if stream:
            return StreamingResponse(
                stream_chat_response(model, messages), media_type="text/event-stream"
            )

        # Create a mock response for demonstration
        response = ChatCompletionResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": "This is a mock response from the OpenAI API implementation.",
                    },
                    finish_reason=OpenAIFinishReason.STOP,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, tokens_per_second=1000),
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_response(model: str, messages: List):
    """
    Stream chat completion response in SSE format.
    """
    response_id = f"chatcmpl-{str(uuid.uuid4())}"

    # Initial chunk
    chunk = ChatCompletionChunkResponse(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    )
    yield f"data: {chunk.json(exclude_none=True)}\n\n"

    # Content chunks - simulating streaming response
    content = "This is a mock streaming response from the OpenAI API implementation."
    for word in content.split():
        chunk = ChatCompletionChunkResponse(
            id=response_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[
                {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
            ],
        )
        yield f"data: {chunk.json(exclude_none=True)}\n\n"
        await asyncio.sleep(0.1)  # Simulate processing time

    # Final chunk
    chunk = ChatCompletionChunkResponse(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
    )
    yield f"data: {chunk.json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: Request, completion_request: CompletionRequest):
    """
    Create a completion based on the provided prompt.
    """
    try:
        model = completion_request.model
        prompt = completion_request.prompt
        stream = completion_request.stream or False

        if stream:
            return StreamingResponse(
                stream_completion_response(model, prompt),
                media_type="text/event-stream",
            )

        # Create a mock response for demonstration
        response = CompletionResponse(
            id=f"cmpl-{str(uuid.uuid4())}",
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=[
                CompletionChoice(
                    index=0,
                    text="This is a mock completion response.",
                    finish_reason=OpenAIFinishReason.STOP,
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, tokens_per_second=1000),
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_completion_response(model: str, prompt):
    """
    Stream completion response in SSE format.
    """
    response_id = f"cmpl-{str(uuid.uuid4())}"

    # Content chunks - simulating streaming response
    content = "This is a mock streaming completion response."
    for word in content.split():
        chunk = {
            "id": response_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": word + " ",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)  # Simulate processing time

    # Final chunk
    chunk = {
        "id": response_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: Request, embedding_request: EmbeddingRequest):
    """
    Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.
    """
    try:
        model = embedding_request.model
        input_text = embedding_request.input

        # Create a mock embedding (a vector of 1536 dimensions is typical for OpenAI embeddings)
        embedding_dim = 1536
        mock_embedding = [0.0] * embedding_dim

        # Set a few random values to make it look more realistic
        import random

        for i in range(10):
            idx = random.randint(0, embedding_dim - 1)
            mock_embedding[idx] = random.uniform(-1, 1)

        response = EmbeddingResponse(
            object="list",
            data=[{"object": "embedding", "embedding": mock_embedding, "index": 0}],
            model=model,
            usage=Usage(prompt_tokens=10, completion_tokens=0, tokens_per_second=1000),
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/moderations", response_model=ModerationResponse)
async def create_moderation(request: Request, moderation_request: ModerationRequest):
    """
    Classifies if text is potentially harmful or violates OpenAI's usage policies.
    """
    try:
        input_text = moderation_request.input
        model = moderation_request.model or "text-moderation-latest"

        # Always return safe content for the mock implementation
        response = ModerationResponse(
            id=f"modr-{str(uuid.uuid4())}",
            model=model,
            results=[
                ModerationResult(
                    categories={
                        "hate": False,
                        "hate_threatening": False,
                        "self_harm": False,
                        "sexual": False,
                        "sexual_minors": False,
                        "violence": False,
                        "violence_graphic": False,
                    },
                    category_scores={
                        "hate": 0.0,
                        "hate_threatening": 0.0,
                        "self_harm": 0.0,
                        "sexual": 0.0,
                        "sexual_minors": 0.0,
                        "violence": 0.0,
                        "violence_graphic": 0.0,
                    },
                    flagged=False,
                )
            ],
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(request: Request, image_request: ImageGenerationRequest):
    """
    Creates an image given a prompt.
    """
    try:
        prompt = image_request.prompt
        n = image_request.n or 1

        # Mock image URL
        mock_url = "https://example.com/generated-image.png"

        response = ImageGenerationResponse(
            created=int(time.time()), data=[ImageData(url=mock_url) for _ in range(n)]
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/edits", response_model=ImageEditResponse)
async def edit_image(request: Request, edit_request: ImageEditRequest):
    """
    Creates an edited or extended image given an original image and a prompt.
    """
    try:
        image = edit_request.image
        prompt = edit_request.prompt
        n = edit_request.n or 1

        # Mock image URL
        mock_url = "https://example.com/edited-image.png"

        response = ImageEditResponse(
            created=int(time.time()), data=[ImageData(url=mock_url) for _ in range(n)]
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/variations", response_model=ImageVariationResponse)
async def create_image_variation(
    request: Request, variation_request: ImageVariationRequest
):
    """
    Creates a variation of a given image.
    """
    try:
        image = variation_request.image
        n = variation_request.n or 1

        # Mock image URL
        mock_url = "https://example.com/variation-image.png"

        response = ImageVariationResponse(
            created=int(time.time()), data=[ImageData(url=mock_url) for _ in range(n)]
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/audio/transcriptions")
async def create_transcription(request: Request):
    """
    Transcribes audio into the input language.
    """
    return JSONResponse({"text": "This is a mock transcription of the provided audio."})


@router.post("/v1/audio/translations")
async def create_translation(request: Request):
    """
    Translates audio into English.
    """
    return JSONResponse(
        {"text": "This is a mock translation of the provided audio into English."}
    )


@router.post("/v1/vision/completions", response_model=VisionResponse)
async def vision_completion(request: Request, vision_request: VisionRequest):
    """
    Analyzes images provided in the messages.
    """
    try:
        model = vision_request.model
        messages = vision_request.messages
        detail = vision_request.detail or OpenAIImageDetail.AUTO

        response = VisionResponse(
            id=f"vision-{str(uuid.uuid4())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": "I can see an image in your message. This is a mock vision analysis response.",
                    },
                    finish_reason=OpenAIFinishReason.STOP,
                )
            ],
            usage=Usage(
                prompt_tokens=100, completion_tokens=30, tokens_per_second=1000
            ),
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
