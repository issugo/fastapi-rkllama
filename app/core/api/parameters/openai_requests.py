from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

from core.api.parameters.commons import Message, Tool, ToolChoice
from core.api.parameters.openai_commons import (
    OpenAIResponseFormat,
    OpenAIImageDetail,
    OpenAIResponseFormatOption,
)


class ChatCompletionRequest(BaseModel):
    """Request for chat completion from OpenAI API."""

    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(
        ..., description="List of messages in the conversation"
    )
    frequency_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify likelihood of token generation"
    )
    logprobs: Optional[bool] = Field(
        None, description="Whether to return log probabilities"
    )
    top_logprobs: Optional[int] = Field(
        None, description="Number of most likely tokens to return"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )
    n: Optional[int] = Field(
        None, gt=0, description="Number of chat completion choices to generate"
    )
    presence_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Presence penalty"
    )
    response_format: Optional[OpenAIResponseFormat] = Field(
        None, description="Format for the response"
    )
    seed: Optional[int] = Field(None, description="Seed for deterministic generation")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Sequences where the API will stop generating"
    )
    stream: Optional[bool] = Field(None, description="Whether to stream the response")
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    tools: Optional[List[Tool]] = Field(
        None, description="List of tools the model may call"
    )
    tool_choice: Optional[ToolChoice] = Field(
        None, description="Controls which tool is called by the model"
    )
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")


class CompletionRequest(BaseModel):
    """Request for text completion from OpenAI API."""

    model: str = Field(..., description="ID of the model to use")
    prompt: Union[str, List[str], List[List[str]]] = Field(
        ..., description="Prompt to generate completions for"
    )
    best_of: Optional[int] = Field(
        None, gt=0, description="Number of completions to generate server-side"
    )
    echo: Optional[bool] = Field(None, description="Whether to echo back the prompt")
    frequency_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        None, description="Modify likelihood of token generation"
    )
    logprobs: Optional[int] = Field(
        None, description="Include log probabilities of most likely tokens"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )
    n: Optional[int] = Field(
        None, gt=0, description="Number of completions to generate"
    )
    presence_penalty: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Presence penalty"
    )
    seed: Optional[int] = Field(None, description="Seed for deterministic generation")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Sequences where the API will stop generating"
    )
    stream: Optional[bool] = Field(None, description="Whether to stream the response")
    suffix: Optional[str] = Field(
        None, description="Suffix to append to the completion"
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")


class EmbeddingRequest(BaseModel):
    """Request for embedding generation from OpenAI API."""

    model: str = Field(..., description="ID of the model to use")
    input: Union[str, List[str], List[int], List[List[int]]] = Field(
        ..., description="Input to generate embeddings for"
    )
    encoding_format: Optional[str] = Field(
        None, description="Format of the output embeddings"
    )
    dimensions: Optional[int] = Field(
        None, description="Number of dimensions for the embeddings"
    )
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")


class ModerationRequest(BaseModel):
    """Request for content moderation from OpenAI API."""

    input: Union[str, List[str]] = Field(..., description="Text to moderate")
    model: Optional[str] = Field(
        None, description="ID of the model to use for moderation"
    )


class ImageGenerationRequest(BaseModel):
    """Request for image generation from OpenAI API."""

    prompt: str = Field(..., description="Text description of the desired image")
    model: Optional[str] = Field(None, description="ID of the model to use")
    n: Optional[int] = Field(
        None, gt=0, le=10, description="Number of images to generate"
    )
    quality: Optional[str] = Field(None, description="Quality of the generated images")
    response_format: Optional[str] = Field(
        None, description="Format of the generated images"
    )
    size: Optional[str] = Field(None, description="Size of the generated images")
    style: Optional[str] = Field(None, description="Style of the generated images")
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")


class ImageEditRequest(BaseModel):
    """Request for image editing from OpenAI API."""

    image: str = Field(..., description="Image to edit")
    prompt: str = Field(..., description="Text description of the desired edit")
    mask: Optional[str] = Field(None, description="Mask image")
    model: Optional[str] = Field(None, description="ID of the model to use")
    n: Optional[int] = Field(
        None, gt=0, le=10, description="Number of images to generate"
    )
    size: Optional[str] = Field(None, description="Size of the generated images")
    response_format: Optional[str] = Field(
        None, description="Format of the generated images"
    )
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")


class ImageVariationRequest(BaseModel):
    """Request for image variation from OpenAI API."""

    image: str = Field(..., description="Image to use as the basis for variations")
    model: Optional[str] = Field(None, description="ID of the model to use")
    n: Optional[int] = Field(
        None, gt=0, le=10, description="Number of images to generate"
    )
    response_format: Optional[str] = Field(
        None, description="Format of the generated images"
    )
    size: Optional[str] = Field(None, description="Size of the generated images")
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")


class VisionRequest(BaseModel):
    """Request for vision analysis from OpenAI API."""

    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(
        ..., description="List of messages including images for analysis"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    stream: Optional[bool] = Field(None, description="Whether to stream the response")
    detail: Optional[OpenAIImageDetail] = Field(
        None, description="Detail level for image analysis"
    )
    user: Optional[str] = Field(None, description="Unique identifier for the end-user")
    response_format: Optional[OpenAIResponseFormatOption] = Field(
        None, description="Format for structured responses"
    )
