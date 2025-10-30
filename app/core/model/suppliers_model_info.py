import json
from pathlib import Path
from typing import List, Optional, Dict, Annotated, Any

from pydantic import BaseModel, Field, BeforeValidator

from core.model import logger
from core.model.ModelLicense import ModelLicense
from core.model.OllamaManifest import OllamaManifest
from core.model.SupplierModelInfo import SupplierModelInfo
from core.model.storage_helpers.SupplierFileInfo import Supplier


class OllamaModelDetails(SupplierModelInfo):
    model_format: str = Field(description="ex: gguf")
    model_family: str = Field(description="ex: llama")


class OllamaRootfs(BaseModel):
    type: str
    diff_ids: List[str]  # list of sha256 to model, license & system files

class OllamaModelLicense(ModelLicense):
    digest: str

    @staticmethod
    def from_content(content: str, license_url: str, digest: str):
        model_license: ModelLicense = ModelLicense.from_content(content)
        return OllamaModelLicense(
            supplier=Supplier.OLLAMA,
            license_name=model_license.common_license,
            license_url=license_url,
            license_text=model_license.license_text,
            digest=digest
        )

class OllamaModelInfo(OllamaModelDetails):
    model_families: List[str]
    model_type: str  # parameter size
    file_type: str  # quantization level
    architecture: str  # ex: amd64, from processor
    os: Optional[str] = "Linux"
    rootfs: OllamaRootfs

    _license: Optional[OllamaModelLicense] = None
    _ollama_manifest: Optional[OllamaManifest] = None

    @property
    def supplier(self):
        return Supplier.OLLAMA

    @property
    def ollama_manifest(self):
        return self._ollama_manifest

    @ollama_manifest.setter
    def ollama_manifest(self, ollama_manifest: OllamaManifest):
        self._ollama_manifest = ollama_manifest

    @property
    def license(self):
        return self._license

    @license.setter
    def license(self, value: OllamaModelLicense):
        self._license = value

    @property
    def template(self):
        if self.ollama_manifest:
            return self.ollama_manifest.template
        else:
            return None

    @property
    def system_prompt(self):
        if self.ollama_manifest:
            return self.ollama_manifest.system
        else:
            return None

    @classmethod
    def load(cls, file_path: str | Path):
        with open(file_path, "r") as f:
            return OllamaModelInfo(**json.load(f))

    def save(self, file_path: str | Path):
        """ write in <MODELS>/blobs/sha256-<DIGEST>"""
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2, by_alias=True))


class HFModelConfig(BaseModel):
    architectures: List[str]
    model_type: str
    tokenizer_config: Dict[str, Optional[str]]
    chat_template_jinja: Optional[str] = None


class HFCardData(BaseModel):
    base_model: List[str]
    tags: List[str]
    params: int


class HFSibling(BaseModel):
    rfilename: str


PyObjectId = Annotated[str, BeforeValidator(str)]

class HFModelLicense(ModelLicense):
    @staticmethod
    def from_content(content: str, license_url: str):
        model_license: ModelLicense = ModelLicense.from_content(content)
        return HFModelLicense(
            supplier=Supplier.HUGGINGFACE,
            license_name=model_license.common_license,
            license_url=license_url,
            license_text=model_license.license_text
        )

class HFModelInfo(SupplierModelInfo):
    hf_id: Optional[PyObjectId] = Field(alias='_id', default=None)
    id: str
    private: bool
    tags: List[str]
    downloads: int
    likes: int
    modelId: str
    author: str
    sha: str
    lastModified: str
    gated: bool
    disabled: bool
    model_index: Optional[Any] = None
    config: HFModelConfig
    cardData: HFCardData
    siblings: List[HFSibling]
    spaces: List[Any] = []
    createdAt: str
    usedStorage: int
    languages: List[str]
    description: Optional[str] = None

    _license: Optional[HFModelLicense] = None

    @property
    def supplier(self):
        return Supplier.HUGGINGFACE

    @property
    def license(self):
        return self._license

    @license.setter
    def license(self, value: HFModelLicense):
        self._license = value

    @property
    def template(self):
        if self.config.chat_template_jinja:
            return self.config.chat_template_jinja
        else:
            return None

    @classmethod
    def load(cls, file_path: str | Path):
        logger.debug(f"HFModelInfo.load(file_path={file_path})")
        with open(file_path, "r") as f:
            return HFModelInfo(**json.load(f))

    def save(self, file_path: str| Path):
        logger.debug(f"HFModelInfo.save(file_path={file_path})")
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2, by_alias=True))
