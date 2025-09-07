from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl


class Role(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageContent(BaseModel):
    """Base class for message content."""
    type: str = Field(..., description="Type of message content")


class TextContent(MessageContent):
    """Text content in a message."""
    type: str = Field("text", const=True)
    text: str = Field(..., description="Text content")


class ImageContent(MessageContent):
    """Image content in a message."""
    type: str = Field("image", const=True)
    image_url: HttpUrl = Field(..., description="URL of the image")


class Message(BaseModel):
    """Message in a conversation."""
    role: Role
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None


class ToolCall(BaseModel):
    """Base class for tool calls."""
    type: str = Field(..., description="Type of tool call")


class FunctionCall(BaseModel):
    """Function call parameters."""
    name: str = Field(..., description="Name of the function to call")
    arguments: str = Field(..., description="JSON-encoded arguments for the function")


class FunctionToolCall(ToolCall):
    """Function tool call."""
    type: str = Field("function", const=True)
    function: FunctionCall = Field(..., description="Function call details")
    id: str = Field(..., description="Unique ID of the tool call")


class ToolChoice(BaseModel):
    """Tool choice configuration."""
    type: str = Field(..., description="Type of tool choice")


class ModelParameters(BaseModel):
    """Common model parameters."""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum number of tokens to generate")
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: int


class Tool(BaseModel):
    """Tool definition."""
    type: str = Field(..., description="Type of tool")


class FunctionTool(Tool):
    """Function tool definition."""
    type: str = Field("function", const=True)
    function: Dict[str, Any] = Field(..., description="Function definition")

class ContentType(str, Enum):
    """Content type for messages."""
    TEXT = "text"
    IMAGE = "image"
    JSON = "json"


class ToolCall(BaseModel):
    """Base tool call model."""
    id: str
    type: str = "function"
    function: Dict[str, Any]


class Tool(BaseModel):
    """Base tool definition."""
    type: str = "function"
    function: Dict[str, Any]
