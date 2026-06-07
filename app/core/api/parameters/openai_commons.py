from enum import Enum
from typing import Dict, Optional, Any, Literal
from pydantic import BaseModel, Field

from core.model.Model import Model

OPENAI_OWNER = "openai"


class OpenAIRoleEnum(str, Enum):
    """Role in an OpenAI message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class OpenAIFinishReason(str, Enum):
    """Reason why the model stopped generating tokens."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    FUNCTION_CALL = "function_call"


class OpenAIResponseFormat(BaseModel):
    """Format for OpenAI response."""

    type: str = Field(..., description="Type of the response format")


class OpenAIJSONResponseFormat(OpenAIResponseFormat):
    """JSON format for OpenAI response."""

    type: str = Literal["json_object"]


class OpenAIResponseFormatOption(BaseModel):
    """Schema for structured output format."""

    type: str = Field(..., description="Type of the format")
    schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema definition")


class OpenAIImageDetail(str, Enum):
    """Detail level for image analysis."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class OpenAIContentFilter(BaseModel):
    """Content filter results."""

    hate: bool = Field(
        False, description="Whether the content was filtered due to hate"
    )
    hate_threatening: bool = Field(
        False, description="Whether the content was filtered due to threatening hate"
    )
    self_harm: bool = Field(
        False, description="Whether the content was filtered due to self-harm"
    )
    sexual: bool = Field(
        False, description="Whether the content was filtered due to sexual content"
    )
    sexual_minors: bool = Field(
        False,
        description="Whether the content was filtered due to sexual content involving minors",
    )
    violence: bool = Field(
        False, description="Whether the content was filtered due to violence"
    )
    violence_graphic: bool = Field(
        False, description="Whether the content was filtered due to graphic violence"
    )


class OpenAIChoice(BaseModel):
    """Base class for choices in OpenAI responses."""

    index: int
    finish_reason: Optional[OpenAIFinishReason] = None


class OpenAIModel(BaseModel):
    id: str = Field(description="model name like gpt-3.5-turbo")
    object: str = "model"
    created: int = Field(description="date as int format, like 1677610602")
    owned_by: str = Field(default=OPENAI_OWNER, description="owner of the model")

    @staticmethod
    def from_model(model: Model):
        return OpenAIModel(
            id=model.id.replace("/", ":"),
            created=int(model.model_info.created_at_dt.timestamp()),
            owned_by=(
                model.model_info.author if model.model_info.author else OPENAI_OWNER
            ),
        )
