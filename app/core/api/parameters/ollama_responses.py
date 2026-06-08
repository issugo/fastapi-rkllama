from typing import List, Optional
from pydantic import BaseModel, Field

from core.api.parameters.commons import Message
from core.api.parameters.ollama_commons import OllamaModelInfo, OllamaModelInfoDetails


class OllamaGenerateResponse(BaseModel):
    """Response from a generate request."""

    model: str = Field(..., description="Name of the model used")
    created_at: str = Field(..., description="Creation time")
    response: str = Field(..., description="Generated response")
    done: bool = Field(..., description="Whether generation is complete")
    context: Optional[List[int]] = Field(None, description="Context token IDs")
    total_duration: Optional[int] = Field(
        None, description="Total duration in nanoseconds"
    )
    load_duration: Optional[int] = Field(
        None, description="Load duration in nanoseconds"
    )
    prompt_eval_duration: Optional[int] = Field(
        None, description="Prompt evaluation duration in nanoseconds"
    )
    eval_duration: Optional[int] = Field(
        None, description="Evaluation duration in nanoseconds"
    )
    eval_count: Optional[int] = Field(None, description="Number of tokens evaluated")
    prompt_eval_count: Optional[int] = Field(
        None, description="Number of prompt tokens evaluated"
    )


class OllamaChatResponse(BaseModel):
    """Response from a chat request."""

    model: str = Field(..., description="Name of the model used")
    created_at: str = Field(..., description="Creation time")
    message: Message = Field(..., description="Response message")
    done: bool = Field(..., description="Whether generation is complete")
    total_duration: Optional[int] = Field(
        None, description="Total duration in nanoseconds"
    )
    load_duration: Optional[int] = Field(
        None, description="Load duration in nanoseconds"
    )
    prompt_eval_duration: Optional[int] = Field(
        None, description="Prompt evaluation duration in nanoseconds"
    )
    eval_count: Optional[int] = Field(None, description="Number of tokens evaluated")
    prompt_eval_count: Optional[int] = Field(
        None, description="Number of prompt tokens evaluated"
    )


class OllamaEmbeddingResponse(BaseModel):
    """Response from an embedding request."""

    embedding: List[float] = Field(..., description="The embedding vector")


class OllamaListResponse(BaseModel):
    """Response from a list models request."""

    models: List[OllamaModelInfo] = Field(..., description="List of models")


class OllamaModelShowDetails(OllamaModelInfoDetails):
    parent_model: str = Field(default="", description="huggingface_path")
    families: List[str] = Field(description="list of families")


class OllamaHFModelShow(BaseModel):
    """Response from a show model request."""

    repo_id: str = Field(..., description="huggingface_path")
    description: str = Field(default="", description="huggingface_description")
    tags: list[str] = Field(default=[], description="huggingface_tags")
    downloads: int = Field(default=0, description="huggingface download count")
    likes: int = Field(default=0, description="huggingface download count")


class OllamaShowResponse(OllamaModelInfo):
    """Response from a show model request."""

    license: str = Field(default="Unknown")
    modelfile: str = Field(default="", description="Contents of the Modelfile")
    parameters: str = Field(description="Parameters of the model")
    template: str = Field(default="", description="Template of the model")
    system: str = Field(default="", description="System prompt of the model")
    details: OllamaModelShowDetails = Field(..., description="Model details")
    model_info: dict = Field(..., description="Model details")
    capabilities: list[str] = Field(
        ..., description="Model capabilities, like completion, tools, ..."
    )
    huggingface: Optional[OllamaHFModelShow] = Field(
        None, description="Huggingface metadata"
    )


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


class OllamaProcessModel(OllamaModelInfo):
    """Model information for a running process."""

    expires_at: str = Field(..., description="Expiration time")
    size_vram: int = Field(..., description="Size in VRAM")


class OllamaPsResponse(BaseModel):
    """Response from a ps request."""

    models: List[OllamaProcessModel] = Field(..., description="List of running models")
