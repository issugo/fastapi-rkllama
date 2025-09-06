from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from app.core.parameters.commons import Message, Usage
from app.core.parameters.ollama_commons import OllamaModelInfo


class OllamaGenerateResponse(BaseModel):
    """Response from a generate request."""
    model: str = Field(..., description="Name of the model used")
    created_at: str = Field(..., description="Creation time")
    response: str = Field(..., description="Generated response")
    done: bool = Field(..., description="Whether generation is complete")
    context: Optional[List[int]] = Field(None, description="Context token IDs")
    total_duration: Optional[int] = Field(None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Load duration in nanoseconds")
    prompt_eval_duration: Optional[int] = Field(None, description="Prompt evaluation duration in nanoseconds")
    eval_duration: Optional[int] = Field(None, description="Evaluation duration in nanoseconds")
    eval_count: Optional[int] = Field(None, description="Number of tokens evaluated")
    prompt_eval_count: Optional[int] = Field(None, description="Number of prompt tokens evaluated")


class OllamaChatResponse(BaseModel):
    """Response from a chat request."""
    model: str = Field(..., description="Name of the model used")
    created_at: str = Field(..., description="Creation time")
    message: Message = Field(..., description="Response message")
    done: bool = Field(..., description="Whether generation is complete")
    total_duration: Optional[int] = Field(None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(None, description="Load duration in nanoseconds")
    prompt_eval_duration: Optional[int] = Field(None, description="Prompt evaluation duration in nanoseconds")
    eval_count: Optional[int] = Field(None, description="Number of tokens evaluated")
    prompt_eval_count: Optional[int] = Field(None, description="Number of prompt tokens evaluated")


class OllamaEmbeddingResponse(BaseModel):
    """Response from an embedding request."""
    embedding: List[float] = Field(..., description="The embedding vector")


class OllamaListResponse(BaseModel):
    """Response from a list models request."""
    models: List[OllamaModelInfo] = Field(..., description="List of models")


class OllamaShowResponse(OllamaModelInfo):
    """Response from a show model request."""
    pass


class OllamaPullResponse(BaseModel):
    """Response from a pull model request."""
    status: str = Field(..., description="Status message")
    digest: Optional[str] = Field(None, description="Model digest")
    total: Optional[int] = Field(None, description="Total size in bytes")
    completed: Optional[int] = Field(None, description="Completed size in bytes")


class OllamaPushResponse(BaseModel):
    """Response from a push model request."""
    status: str = Field(..., description="Status message")
    digest: Optional[str] = Field(None, description="Model digest")
    total: Optional[int] = Field(None, description="Total size in bytes")
    completed: Optional[int] = Field(None, description="Completed size in bytes")


class OllamaCreateResponse(BaseModel):
    """Response from a create model request."""
    status: str = Field(..., description="Status message")


class OllamaCopyResponse(BaseModel):
    """Response from a copy model request."""
    status: str = Field(..., description="Status message")


class OllamaDeleteResponse(BaseModel):
    """Response from a delete model request."""
    status: str = Field(..., description="Status message")
