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

    def __init__(
        self,
        model_path: ModelPath,
        sha256_digest: str,
        generic_model_file: ModelFile,
        logger=pkg_logger,
    ):
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
            digest = digest[len(BLOB_PREFIX) :]
        return os.path.join(
            OllamaModelStorageHelper.blobs_dir(), f"{BLOB_PREFIX}{digest}"
        )

    @classmethod
    def store_blob_link(cls, blob_link: str | Path, digest: str):
        if isinstance(blob_link, str):
            blob_link = Path(blob_link)
        if blob_link.exists():
            if not os.path.islink(blob_link):
                raise Exception(f"Blob link {blob_link} exists but is not a symlink")
        else:
            root_common_path = Path(get_settings().paths.models)
            if not root_common_path.exists():
                root_common_path.mkdir(parents=True)
            if not blob_link.parent.exists():
                blob_link.parent.mkdir(parents=True)
            target_file_path = cls.blob_path(digest=digest)
            blob_link.symlink_to(
                cls.build_relative_link_path(
                    target_file_path=target_file_path,
                    link_path=str(blob_link),
                    root_common_path=f"{str(root_common_path)}/",
                )
            )

    @property
    def model_blob_path(self) -> Tuple[str, str]:
        if self._model_blob_path is None:
            self._model_blob_path = OllamaModelStorageHelper.blob_path(
                digest=self.sha256_digest
            )
        return self._model_blob_path, self.sha256_digest

    def clean(
        self, generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo
    ):
        if self.model_path:
            ModelFile.clean(self.model_path)
        elif generic_model_file_info:
            ModelFile.clean(generic_model_file_info)

    def store(self):
        if self.generic_model_file and self.model_path:
            self.generic_model_file.save()
