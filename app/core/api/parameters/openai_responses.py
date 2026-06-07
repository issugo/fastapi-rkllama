from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from core.api.parameters.commons import Message, Usage
from core.api.parameters.openai_commons import (
    OpenAIChoice,
    OpenAIContentFilter,
)


class ChatCompletionChoice(OpenAIChoice):
    """Choice in a chat completion response."""

    message: Message
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionChunkChoice(OpenAIChoice):
    """Choice in a chat completion chunk response."""

    delta: Dict[str, Any]
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion request."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkResponse(BaseModel):
    """Response chunk from a streaming chat completion request."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None


class CompletionChoice(OpenAIChoice):
    """Choice in a completion response."""

    text: str
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """Response from a completion request."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class Embedding(BaseModel):
    """Single embedding vector."""

    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Usage


class ModerationCategoryScores(BaseModel):
    """Scores for each moderation category."""

    hate: float
    hate_threatening: float
    self_harm: float
    sexual: float
    sexual_minors: float
    violence: float
    violence_graphic: float


class ModerationResult(BaseModel):
    """Result of a moderation."""

    categories: OpenAIContentFilter
    category_scores: ModerationCategoryScores
    flagged: bool


class ModerationResponse(BaseModel):
    """Response from a moderation request."""

    id: str
    model: str
    results: List[ModerationResult]


class ImageData(BaseModel):
    """Data for a generated image."""

    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """Response from an image generation request."""

    created: int
    data: List[ImageData]


class ImageEditResponse(BaseModel):
    """Response from an image edit request."""

    created: int
    data: List[ImageData]


class ImageVariationResponse(BaseModel):
    """Response from an image variation request."""

    created: int
    data: List[ImageData]


class ToolResponse(BaseModel):
    """Response from a tool call."""

    id: str = Field(..., description="ID of the tool call")
    type: str = Field(..., description="Type of the tool")
    function: Dict[str, Any] = Field(..., description="Function details")


class AssistantMessageToolCall(BaseModel):
    """Tool call in an assistant message."""

    id: str = Field(..., description="ID of the tool call")
    type: str = Field(..., description="Type of the tool call")
    function: Dict[str, Any] = Field(..., description="Function details")


class AssistantMessage(BaseModel):
    """Message from an assistant."""

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[AssistantMessageToolCall]] = None


class VisionResponse(BaseModel):
    """Response from a vision request."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None
