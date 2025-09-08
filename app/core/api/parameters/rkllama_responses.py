from typing import Optional, List

from pydantic import BaseModel

from core.api.parameters import Usage


class RKllamaUsage(Usage):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    total_tokens: int = 0

class RKllamaChoice(BaseModel):
    """Base class for choices in RKllama responses."""
    role: str = "assistant"
    content: str =""
    logprobs: Optional[str] = None
    finish_reason: str = "stop"


class RKllamaResponse(BaseModel):
    id: str = "rkllm_chat"
    object: str = "rkllm_chat"
    created: int = 0
    choices: List[RKllamaChoice] = [RKllamaChoice()]
    usage: RKllamaUsage = RKllamaUsage()
