import os
from typing import Optional

from pydantic import BaseModel
from safetensors import torch


class ConversionConfig(BaseModel):
    """
    model_id: Hugging Face model ID
    output-dir: Output directory for converted models, default='data/models
    quantization: Quantization format (Q4_0, Q4_K_M, Q8_0, Q8_K_M), default='Q4_0'
    max-context-len: Maximum context length, type=int, default=4096
    dtype: Model data type (float16 or float32), default='float16'
    device: Device to use for conversion, default='cuda' if torch.cuda.is_available() else 'cpu'
    token: Hugging Face token for private models
    """
    model_id: str
    output_dir: str
    quantization: str = 'Q4_0'
    max_context_len: int = 4096
    dtype: str = 'float16'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    token: Optional[str] = os.getenv("HF_TOKEN")  # Get token from environment variable

    @property
    def model_name(self) -> str:
        """Get model name from model ID."""
        return self.model_id.split('/')[-1]

    @property
    def output_path(self) -> str:
        """Get the full output path including model name."""
        return os.path.join(self.output_dir, self.model_name)
