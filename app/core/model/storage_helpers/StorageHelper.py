import os.path
from pathlib import Path

from core.model.ModelFile import ModelFile, ModelFileInfo
from core.model.storage_helpers import logger as pkg_logger


class StorageHelper:
    ollama_model_storage_helper = None
    logger = None

    def __init__(self, ollama_model_storage_helper):
        self.ollama_model_storage_helper = ollama_model_storage_helper
        if ollama_model_storage_helper.logger:
            self.logger = ollama_model_storage_helper.logger
        else:
            self.logger = pkg_logger

    @staticmethod
    def build_relative_link_path(target_file_path: str, link_path: str, root_common_path: str) -> Path:
        """
        Builds a relative link path from a target file path, a link path, and a root common path.
        """
        pkg_logger.debug(f"build_relative_link_path(): target_file_path={target_file_path}, link_path={link_path}, root_common_path={root_common_path}")

        if not target_file_path:
            raise ValueError("target_file_path cannot be empty")
        if not link_path:
            raise ValueError("link_path cannot be empty")
        if not root_common_path:
            raise ValueError("root_common_path cannot be empty")
        if not root_common_path.endswith("/"):
            raise ValueError("root_common_path must end with /")
        if not target_file_path.startswith(root_common_path):
            raise ValueError(f"target_file_path must start with root_common_path: {root_common_path}")
        if not link_path.startswith(root_common_path):
            raise ValueError(f"link_path must start with root_common_path: {root_common_path}")
        current_file = Path(link_path).name
        current_dir = Path(link_path.removesuffix(current_file))
        target_file = Path(target_file_path)
        return target_file.relative_to(current_dir, walk_up=True)

    @property
    def model_link(self) -> Path:
        raise NotImplementedError()

    def _store_model_link(self):
        model_link = self.model_link
        if os.path.exists(model_link):
            if not os.path.islink(model_link):
                raise Exception(f"Model link {model_link} exists but is not a symlink")
        else:
            from core.config import config_utils
            root_common_path = Path(config_utils.rkllama_config.paths.models)
            if not root_common_path.exists():
                root_common_path.mkdir(parents=True)
            if not model_link.parent.exists():
                model_link.parent.mkdir(parents=True)
            target_file_path, _ = self.ollama_model_storage_helper.model_blob_path
            model_link.symlink_to(self.build_relative_link_path(
                    target_file_path=target_file_path,
                    link_path=str(model_link),
                    root_common_path=f"{str(root_common_path)}/"
                ))

    def store(self):
        raise NotImplementedError

    def clean(self, generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo):
        raise NotImplementedError
