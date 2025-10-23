import json
import os.path
from pathlib import Path
from typing import Optional, List, Tuple, Any

import requests
from pydantic import BaseModel

"""
sample for devstrall:latest

{
    "schemaVersion":2,
    "mediaType":"application/vnd.docker.distribution.manifest.v2+json",
    "config":{
        "mediaType":"application/vnd.docker.container.image.v1+json",
        "digest":"sha256:3dc762df9951ecae062822bbaa78b01f1252e2e316c911adbedcbe97bbff5b26",
        "size":488
    },
    "layers":[
        {
            "mediaType":"application/vnd.ollama.image.model",
            "digest":"sha256:b3a2c9a8fef9be8d2ef951aecca36a36b9ea0b70abe9359eab4315bf4cd9be01",
            "size":14333909728,
            "from":"devstral:latest"
        },
        {
            "mediaType":"application/vnd.ollama.image.template",
            "digest":"sha256:ea9ec42474e0b11615d2287c0a4b25f89cd1bd84d034113696dfd6af6ba1ae5d",
            "size":823
        },
        {
            "mediaType":"application/vnd.ollama.image.license",
            "digest":"sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            "size":11356,
            "from":"devstral:latest"
        },
        {
            "mediaType":"application/vnd.ollama.image.system",
            "digest":"sha256:5725afc40acd80cbeefba61e41cf50eb7924f6ed2fe6aec2dc6fa0e9f2c396d1",
            "size":5651
        }
    ]
}

"""

VND_OLLAMA_IMAGE_SYSTEM = "application/vnd.ollama.image.system"
VND_OLLAMA_IMAGE_MODEL = "application/vnd.ollama.image.model"


class OllamaManifestLayer(BaseModel):
    mediaType: str
    digest: str
    size: int
    from_: Optional[str] = None

    class Config:
        fields = {
            'from_': 'from'  # Map Python field 'from_' to JSON field 'from'
        }

class OllamaManifestModelLayer(OllamaManifestLayer):
    pass
    # mediaType: str = "application/vnd.ollama.image.model"
    # with from

class OllamaManifestLicenseLayer:
    pass
    # mediaType: str = "application/vnd.ollama.image.license"
    # with from

class OllamaManifestSystemLayer:
    pass
    # mediaType: str = "application/vnd.ollama.image.system"
    # no from


class OllamaManifestConfig(BaseModel):
    mediaType: str = "application/vnd.docker.container.image.v1+json"
    digest: str
    size: int


class OllamaManifest(BaseModel):
    schemaVersion: int = 2
    mediaType: str = "application/vnd.docker.distribution.manifest.v2+json"
    config: OllamaManifestConfig
    layers: List[OllamaManifestLayer]

    _ollama_manifest_model_layer: OllamaManifestModelLayer | None = None

    @property
    def ollama_manifest_model_layer(self):
        if self._ollama_manifest_model_layer is None:
            for layer in self.layers:
                if layer.mediaType == VND_OLLAMA_IMAGE_MODEL:
                    self._ollama_manifest_model_layer = OllamaManifestModelLayer.model_validate_json(layer.model_dump_json())
                    break
        return self._ollama_manifest_model_layer

    @property
    def size(self):
        return int(self.ollama_manifest_model_layer.size)

    @property
    def lfs_sha256(self):
        digest: str = self.ollama_manifest_model_layer.digest
        if digest.startswith("sha256:"):
            return digest[7:]
        raise ValueError(f"digest={digest} is not a sha256 digest")

    @property
    def ollama_model_config_path(self) -> str:
        if self.config:
            from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper
            return str(OllamaStorageHelper.ollama_model_info_path(
                model_path=None,
                ollama_manifest=self
            ))
        raise ValueError(f"missing config in manifest")

    @classmethod
    def create(cls):
        return cls(**{})

    @classmethod
    def load(cls, ollama_manifest_path: str | Path):
        if isinstance(ollama_manifest_path, str):
            ollama_manifest_path = Path(ollama_manifest_path)
        json_string = ollama_manifest_path.read_text()
        return OllamaManifest.model_validate_json(json_string)

    def save(self, ollama_manifest_path: str | Path):
        """ write in <MODELS>/manifests/registry/library/<MODEL_NAME>:<TAG>"""
        with open(ollama_manifest_path, "w") as f:
            f.write(self.model_dump_json(indent=2))


