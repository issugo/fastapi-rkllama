import os.path
from pathlib import Path

from pydantic import BaseModel


class StorageHelper(BaseModel):
    pass

    @staticmethod
    def build_relative_link_path(target_file_path: str, link_path: str, root_common_path: str) ->Path:
        """
        Builds a relative link path from a target file path, a link path, and a root common path.
        """
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
        relative_link_path = ""
        common_path = Path(root_common_path)
        current_dir = Path(link_path).parent
        while current_dir != common_path:
            relative_link_path = f"../{relative_link_path}"
            current_dir = current_dir.parent
        return Path(os.path.join(relative_link_path, target_file_path.replace(root_common_path,"")))


    def store(self):
        raise NotImplementedError

    def clean(self):
        raise NotImplementedError