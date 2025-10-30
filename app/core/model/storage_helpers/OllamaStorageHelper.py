import os
from pathlib import Path

from core.config.config_utils import get_settings
from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelPath import ModelPath
from core.model.storage_helpers import logger
from core.model.storage_helpers.OllamaModelStorageHelper import OllamaModelStorageHelper
from core.model.storage_helpers.StorageHelper import StorageHelper

MANIFESTS = "manifests"

class OllamaStorageHelper(StorageHelper):
    model_file: ModelFile

    _manifest_dir: Path = None
    _manifest_filename: str = None
    _manifest_path: Path = None
    _links_dir: Path = None

    def __init__(self, ollama_model_storage_helper, model_file: ModelFile):
        super().__init__(ollama_model_storage_helper)
        self.model_file = model_file
        self.logger.debug(
            f"OllamaStorageHelper: model_file.ollama_model_info_exists={model_file.ollama_model_info_exists}")
        self.logger.debug(
            f"OllamaStorageHelper: model_file.ollama_file_info_exists={model_file.ollama_file_info_exists}")


    @staticmethod
    def manifests_dir():
        manifests_dir = os.path.join(get_settings().paths.models, MANIFESTS)
        os.makedirs(manifests_dir, exist_ok=True)
        return manifests_dir

    @staticmethod
    def blobs_dir():
        return OllamaModelStorageHelper.blobs_dir()

    @classmethod
    def __model_name(cls, model_path: ModelPath):
        return model_path.model_name

    @property
    def model_name(self):
        return self.__class__.__model_name(self.model_file.model_file_info)

    @classmethod
    def __model_tag(cls, model_path: ModelPath):
        return model_path.endpoint_model_file.replace(model_path.model_type.get_extension(), "")

    @property
    def model_tag(self):
        return self.__class__.__model_tag(self.model_file.model_file_info)

    @classmethod
    def __manifest_dir(cls, model_path: ModelPath):
        return Path(os.path.join(OllamaStorageHelper.manifests_dir(), model_path.model_name))

    @property
    def manifest_dir(self) -> Path:
        if self._manifest_dir is None:
            self._manifest_dir = self.__class__.__manifest_dir(self.model_file.model_file_info)
        return self._manifest_dir

    @classmethod
    def __manifest_filename(cls, model_path: ModelPath):
        return cls.__model_tag(model_path)

    @property
    def manifest_filename(self) -> str:
        if self._manifest_filename is None:
            self._manifest_filename = self.__class__.__manifest_filename(self.model_file.model_file_info)
        return self._manifest_filename

    @classmethod
    def __manifest_path(cls, model_path: ModelPath) -> str:
        return cls.ollama_model_manifest_path(
            model_name=model_path.model_name,
            tag=cls.__model_tag(model_path)
        )

    @property
    def manifest_path(self) -> Path:
        if self._manifest_path is None:
            self._manifest_path = Path(self.__class__.__manifest_path(self.model_file.model_file_info))
        return self._manifest_path

    @staticmethod
    def ollama_model_manifest_path(model_name: str, tag: str) -> str:
        if tag:
            return f"{OllamaStorageHelper.manifests_dir()}/{model_name}/{tag}"
        else:
            raise ValueError("Ollama model tag cannot be empty")

    @staticmethod
    def ollama_file_info_path(model_path: ModelPath) -> Path:
        return Path(OllamaStorageHelper.ollama_model_manifest_path(
            model_name=model_path.model_name,
            tag=model_path.endpoint_model_file.replace(model_path.model_type.get_extension(), "")
        ))

    @staticmethod
    def ollama_model_info_path(model_path: ModelPath, ollama_manifest = None) -> Path | None:
        from core.model.OllamaManifest import OllamaManifest
        from core.model.ModelFile import ModelFileInfo
        try:
            if isinstance(model_path, ModelFileInfo):
                if model_path.ollama_file_info_exists and ollama_manifest is None:
                    ollama_manifest = model_path.ollama_file_info

            if ollama_manifest is None:
                manifest_path: str = OllamaStorageHelper.__manifest_path(model_path=model_path)
                if not os.path.exists(manifest_path):
                    return None
                ollama_manifest: OllamaManifest = OllamaManifest.load(
                    OllamaStorageHelper.__manifest_path(model_path=model_path))
            if ollama_manifest.config:
                digest: str = ollama_manifest.config.digest
                if digest.startswith("sha256:"):
                    ollama_model_info_digest = digest[7:]
                    return Path(OllamaModelStorageHelper.blob_path(digest=ollama_model_info_digest))
                raise ValueError(f"digest={digest} is not a sha256 digest")
        except Exception as e:
            logger.exception(f"Error fetching OLLAMA model info: {str(e)}")
            return None


    @property
    def links_dir(self) -> Path:
        if self._links_dir is None:
            self._links_dir = Path(self.manifest_dir / f".{self.model_tag}")
            os.makedirs(self._links_dir, exist_ok=True)
        return self._links_dir

    @property
    def model_link(self) -> Path:
        return self.links_dir / 'model'

    @property
    def template_link(self) -> Path:
        return self.links_dir / 'template'

    @property
    def system_link(self) -> Path:
        return self.links_dir / 'system'


    def clean(self, generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo):
        try:
            model_link = Path(self.model_link)
            if model_link.exists():
                model_link.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning model link: {str(e)}")

        try:
            ollama_file_info_file_path = OllamaStorageHelper.ollama_file_info_path(self.model_file.model_file_info)
            if ollama_file_info_file_path.exists():
                ollama_file_info_file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning ollama_file_info: {str(e)}")

        try:
            ollama_model_info_file_path=OllamaStorageHelper.ollama_model_info_path(
                model_path=self.model_file.model_file_info,
                ollama_manifest=generic_model_file_info.ollama_file_info)
            if ollama_model_info_file_path.exists():
                ollama_model_info_file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning ollama_model_info: {str(e)}")

    def _store_template_link(self):
        self.ollama_model_storage_helper.store_blob_link(
            blob_link=self.template_link,
            digest=self.model_file.ollama_file_info.ollama_manifest_template_layer.digest
        )

    def _store_system_link(self):
        self.ollama_model_storage_helper.store_blob_link(
            blob_link=self.system_link,
            digest=self.model_file.ollama_file_info.ollama_manifest_system_layer.digest
        )


    def store(self):

        if self.model_file.ollama_model_info_exists:
            file_path: Path = OllamaStorageHelper.ollama_model_info_path(self.model_file.model_file_info)
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True)
            self.model_file.ollama_model_info.save(file_path=file_path)

        if self.model_file.ollama_file_info_exists:
            file_path: Path = OllamaStorageHelper.ollama_file_info_path(self.model_file.model_file_info)
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True)
            self.model_file.ollama_file_info.save(ollama_manifest_path=file_path)

            if self.model_file.ollama_model_info_exists:
                template = self.model_file.ollama_file_info.template
                if template:
                    local_filename = OllamaModelStorageHelper.blob_path(self.model_file.ollama_file_info.ollama_manifest_template_layer.digest)
                    if not Path(local_filename).exists():
                        with open(local_filename, 'wb') as f:
                            f.write(template)
                    self._store_template_link()

                system = self.model_file.ollama_file_info.system
                if system:
                    local_filename = OllamaModelStorageHelper.blob_path(self.model_file.ollama_file_info.ollama_manifest_system_layer.digest)
                    if not Path(local_filename).exists():
                        with open(local_filename, 'wb') as f:
                            f.write(system)
                    self._store_system_link()


        self._store_model_link()

