import os
from pathlib import Path

from core.model.ModelFile import ModelFile
from core.model.ModelPath import ModelPath
from core.model.storage_helpers import logger
from core.model.storage_helpers.OllamaModelStorageHelper import OllamaModelStorageHelper

MANIFESTS = "manifests"


class OllamaStorageHelper(OllamaModelStorageHelper):
    ollama_model_storage_helper: OllamaModelStorageHelper
    model_file: ModelFile

    _manifest_dir: Path = None
    _manifest_filename: str = None
    _manifest_path: Path = None
    _links_dir: Path = None

    @staticmethod
    def manifests_dir():
        from core.config import config_utils
        return os.path.join(config_utils.rkllama_config.paths.models, MANIFESTS)

    @classmethod
    def __model_name(cls, model_path: ModelPath):
        return model_path.model_name

    @property
    def model_name(self):
        return self.__class__.__model_name(self.model_file)

    @classmethod
    def __model_tag(cls, model_path: ModelPath):
        return model_path.endpoint_model_file.replace(model_path.model_type.get_extension(), "")

    @property
    def model_tag(self):
        return self.__class__.__model_tag(self.model_file)

    @classmethod
    def __manifest_dir(cls, model_path: ModelPath):
        return Path(os.path.join(OllamaStorageHelper.manifests_dir(), model_path.model_name))

    @property
    def manifest_dir(self) -> Path:
        if self._manifest_dir is None:
            self._manifest_dir = self.__class__.__manifest_dir(self.model_file)
        return self._manifest_dir

    @classmethod
    def __manifest_filename(cls, model_path: ModelPath):
        return cls.__model_tag(model_path)

    @property
    def manifest_filename(self) -> str:
        if self._manifest_filename is None:
            self._manifest_filename = self.__class__.__manifest_filename(self.model_file)
        return self._manifest_filename

    @classmethod
    def __manifest_path(cls, model_path: ModelPath):
        return cls.__manifest_dir(model_path) / cls.__manifest_filename(model_path)

    @property
    def manifest_path(self) -> Path:
        if self._manifest_path is None:
            self._manifest_path = self.__class__._manifest_path(self.model_file)
        return self._manifest_path

    @staticmethod
    def ollama_model_info_path(model_path: ModelPath) -> Path | None:
        from core.model.OllamaManifest import OllamaManifest
        try:
            ollama_manifest: OllamaManifest = OllamaManifest.load(
                OllamaStorageHelper.__manifest_path(model_path=model_path))
            ollama_model_info_digest = ollama_manifest.config.digest
            return OllamaStorageHelper.__blob_path(ollama_model_info_digest)
        except Exception as e:
            logger.exception(f"Error fetching OLLAMA model info: {str(e)}")
            return None


    @property
    def links_dir(self) -> Path:
        if self._links_dir is None:
            self._links_dir = Path(self.manifest_dir / f".{self.model_tag}")
            os.makedirs(self._links_dir, exist_ok=True)
        return self._links_dir

    def _store_model_link(self):
        model_link = self.links_dir / 'model'
        if os.path.exists(model_link):
            if not os.path.islink(model_link):
                raise Exception(f"Model link {model_link} exists but is not a symlink")
        else:
            from core.config import config_utils
            os.symlink(model_link, self.build_relative_link_path(
                target_file_path=self.ollama_model_storage_helper.model_blob_path,
                link_path=model_link,
                root_common_path=config_utils.rkllama_config.paths.models
            ))

    def store(self):
        self._store_model_link()

        # TODO: search for a system blob with the same content (parse blob links in .systems directory), if not found, create one in blobs, then create its link in .systems directory

        # TODO: generate an OllamaManifest using model blob and system blob, then dump it in manifest_filename

        # dump the model file into the '.'+model_name directory
        self.model_file.save(dotdir)

        raise NotImplementedError