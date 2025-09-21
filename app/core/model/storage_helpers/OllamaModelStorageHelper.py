import os
from pathlib import Path

from core.model.ModelPath import ModelPath
from core.model.storage_helpers.StorageHelper import StorageHelper

BLOBS = "blobs"
BLOB_PREFIX = "sha256-"

class OllamaModelStorageHelper(StorageHelper):
    model_path: ModelPath
    sha256_digest: str

    _model_blob_path = None

    @staticmethod
    def blobs_dir():
        from core.config import config_utils
        return os.path.join(config_utils.rkllama_config.paths.models, BLOBS)

    @classmethod
    def __blob_path(cls, digest: str):
        return os.path.join(cls.blobs_dir(), f"{BLOB_PREFIX}{digest}")

    @property
    def model_blob_path(self):
        if self._model_blob_path is None:
            self._model_blob_path = os.path.join(self.blobs_dir(), f"{BLOB_PREFIX}{self.sha256_digest}")
        return self._model_blob_path
