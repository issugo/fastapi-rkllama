from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class OllamaFormatOption(BaseModel):
    """Schema for structured output format."""
    schema: Dict[str, Any] = Field(..., description="JSON schema definition")


class OllamaOptions(BaseModel):
    """Ollama request options."""
    num_keep: Optional[int] = Field(None, description="Number of tokens to keep from the prompt")
    seed: Optional[int] = Field(None, description="Random seed for generation")
    num_predict: Optional[int] = Field(None, description="Maximum number of tokens to predict")
    top_k: Optional[int] = Field(None, description="Consider top_k most likely tokens")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Consider tokens with top_p probability mass")
    tfs_z: Optional[float] = Field(None, description="TF-IDF normalization parameter Z")
    typical_p: Optional[float] = Field(None, description="Locally typical sampling parameter")
    repeat_last_n: Optional[int] = Field(None, description="Last n tokens to consider for repetition penalty")
    repeat_penalty: Optional[float] = Field(None, description="Penalty for repetition")
    presence_penalty: Optional[float] = Field(None, description="Penalty for presence")
    frequency_penalty: Optional[float] = Field(None, description="Penalty for frequency")
    mirostat: Optional[int] = Field(None, description="Mirostat sampling parameter (0, 1, or 2)")
    mirostat_tau: Optional[float] = Field(None, description="Mirostat target surprise value")
    mirostat_eta: Optional[float] = Field(None, description="Mirostat learning rate")
    penalize_newline: Optional[bool] = Field(None, description="Whether to penalize newlines")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    numa: Optional[bool] = Field(None, description="Whether to use NUMA")
    num_ctx: Optional[int] = Field(None, description="Context window size")
    num_batch: Optional[int] = Field(None, description="Batch size")
    num_thread: Optional[int] = Field(None, description="Number of threads to use")
    num_gpu: Optional[int] = Field(None, description="Number of GPUs to use")
    low_vram: Optional[bool] = Field(None, description="Whether to optimize for low VRAM")
    f16_kv: Optional[bool] = Field(None, description="Whether to use F16 KV cache")
    logits_all: Optional[bool] = Field(None, description="Whether to return logits for all tokens")
    vocab_only: Optional[bool] = Field(None, description="Whether to return vocabulary only")
    use_mmap: Optional[bool] = Field(None, description="Whether to use memory mapping")
    use_mlock: Optional[bool] = Field(None, description="Whether to use mlock")
    embedding_only: Optional[bool] = Field(None, description="Whether to return embedding only")
    rope_frequency_base: Optional[float] = Field(None, description="RoPE frequency base")
    rope_frequency_scale: Optional[float] = Field(None, description="RoPE frequency scale")
    format: Optional[Union[str, OllamaFormatOption]] = Field(None, description="Format for response")


class OllamaModelInfo(BaseModel):
    """Information about an Ollama model."""
    name: str = Field(..., description="Model name")
    modified_at: str = Field(..., description="Last modification time")
    size: int = Field(..., description="Model size in bytes")
    digest: str = Field(..., description="Model digest")
    details: Dict[str, Any] = Field(..., description="Model details")
