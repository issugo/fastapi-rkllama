import os
from pathlib import Path
from typing import Tuple

from core.config.config_utils import get_settings
from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelPath import ModelPath
from core.model.storage_helpers import logger as pkg_logger
from core.model.storage_helpers.StorageHelper import StorageHelper

BLOBS = "blobs"
BLOB_PREFIX = "sha256-"

class OllamaModelStorageHelper(StorageHelper):
    model_path: ModelPath
    sha256_digest: str
    generic_model_file: ModelFile

    _model_blob_path = None

    def __init__(self,
                 model_path: ModelPath, sha256_digest: str,
                 generic_model_file: ModelFile,
                 logger=pkg_logger):
        super().__init__(self)
        self.model_path = model_path
        self.sha256_digest = sha256_digest
        self.generic_model_file = generic_model_file
        self.logger = logger


    @staticmethod
    def blobs_dir():
        blobs_dir = os.path.join(get_settings().paths.models, BLOBS)
        os.makedirs(blobs_dir, exist_ok=True)
        return blobs_dir

    @staticmethod
    def blob_path(digest: str):
        if digest.startswith(BLOB_PREFIX):
            digest = digest[len(BLOB_PREFIX):]
        return os.path.join(OllamaModelStorageHelper.blobs_dir(), f"{BLOB_PREFIX}{digest}")

    @property
    def model_blob_path(self) -> Tuple[str, str]:
        if self._model_blob_path is None:
            self._model_blob_path = OllamaModelStorageHelper.blob_path(digest=self.sha256_digest)
        return self._model_blob_path, self.sha256_digest

    def clean(self, generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo):
        if self.model_path:
            ModelFile.clean(self.model_path)
        elif generic_model_file_info:
            ModelFile.clean(generic_model_file_info)


    def store(self):
        if self.generic_model_file and self.model_path:
            self.generic_model_file.save()



