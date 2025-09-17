from typing import Optional, List

from pydantic import BaseModel

from core.model import OllamaManifest
from core.model.ModelType import ModelType

"""
devstral:latest (from) sample for OllamaModelConfig
config_data = {
    "model_format": "gguf",
    "model_family": "llama",
    "model_families": ["llama"],
    "model_type": "23.6B",
    "file_type": "Q4_K_M",
    "architecture": "amd64",
    "os": "linux",
    "rootfs": {
        "type": "layers",
        "diff_ids": [
            "sha256:b3a2c9a8fef9be8d2ef951aecca36a36b9ea0b70abe9359eab4315bf4cd9be01",
            "sha256:6db27cd4e277c91264572b9c899c1980daa8dea11e902f0070a6f4763f3d13c8",
            "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            "sha256:5725afc40acd80cbeefba61e41cf50eb7924f6ed2fe6aec2dc6fa0e9f2c396d1"
        ]
    }
}

"""

class OllamaModelDetails(BaseModel):
    model_format: Optional[str] # ex: gguf
    model_family: Optional[str] # ex: llama, see

class OllamaRootfs(BaseModel):
    type: str
    diff_ids: List[str] # list of sha256 to model, license & system files


class ModelDetails(OllamaModelDetails):
    parameter_size: str   # ex: 3B
    quantization_level: str

class ModelInfo(BaseModel):
    name: str  # Use simplified name like qwen:3b
    model: str # Match Ollama's format
    modified_at: str
    size: int
    digest: str = "" # Ollama field (sha256 value) (not used but included for compatibility)
    details: ModelDetails
    model_type: ModelType

class OllamaModelInfo(OllamaModelDetails):
    model_families: List[str]
    model_type: str # parameter size
    file_type: str # quantization level
    architecture: str # ex: amd64, from processor
    os: Optional[str] = "Linux"
    rootfs: OllamaRootfs

    @classmethod
    def create(cls):
        return cls(**{})

    @classmethod
    def load(cls, ollama_manifest: OllamaManifest):
        return cls(**{})

    def dump(self):
        """ write in <MODELS>/blobs/sha256-<DIGEST>"""
        pass

