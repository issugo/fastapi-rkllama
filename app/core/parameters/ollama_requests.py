from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

from app.core.parameters.commons import Message
from app.core.parameters.ollama_commons import OllamaOptions, OllamaFormatOption


class OllamaGenerateRequest(BaseModel):
    """Request to generate a response from a prompt."""
    model: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="Prompt to generate a response for")
    options: Optional[OllamaOptions] = Field(None, description="Generation options")
    system: Optional[str] = Field(None, description="System prompt to include")
    template: Optional[str] = Field(None, description="Template to use for generation")
    context: Optional[List[int]] = Field(None, description="Context token IDs")
    stream: Optional[bool] = Field(None, description="Whether to stream the response")
    raw: Optional[bool] = Field(None, description="Whether to return raw response")
    format: Optional[Union[str, OllamaFormatOption]] = Field(None, description="Format for response")


class OllamaChatRequest(BaseModel):
    """Request to chat with a model."""
    model: str = Field(..., description="Name of the model to use")
    messages: List[Message] = Field(..., description="Messages in the chat")
    options: Optional[OllamaOptions] = Field(None, description="Generation options")
    stream: Optional[bool] = Field(None, description="Whether to stream the response")
    format: Optional[Union[str, OllamaFormatOption]] = Field(None, description="Format for response")


class OllamaEmbeddingRequest(BaseModel):
    """Request to generate an embedding."""
    model: str = Field(..., description="Name of the model to use")
    prompt: str = Field(..., description="Text to generate embedding for")
    options: Optional[OllamaOptions] = Field(None, description="Generation options")


class OllamaPullRequest(BaseModel):
    """Request to pull a model."""
    name: str = Field(..., description="Name of the model to pull")
    insecure: Optional[bool] = Field(None, description="Whether to allow insecure connections")
    stream: Optional[bool] = Field(None, description="Whether to stream the response")


class OllamaPushRequest(BaseModel):
    """Request to push a model."""
    name: str = Field(..., description="Name of the model to push")
    insecure: Optional[bool] = Field(None, description="Whether to allow insecure connections")
    stream: Optional[bool] = Field(None, description="Whether to stream the response")


class OllamaCreateRequest(BaseModel):
    """Request to create a model."""
    name: str = Field(..., description="Name of the model to create")
    modelfile: Optional[str] = Field(None, description="Contents of the Modelfile")
    path: Optional[str] = Field(None, description="Path to the Modelfile")
    stream: Optional[bool] = Field(None, description="Whether to stream the response")


class OllamaCopyRequest(BaseModel):
    """Request to copy a model."""
    source: str = Field(..., description="Name of the source model")
    destination: str = Field(..., description="Name of the destination model")


class OllamaDeleteRequest(BaseModel):
    """Request to delete a model."""
    name: str = Field(..., description="Name of the model to delete")
