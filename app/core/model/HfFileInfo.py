import json
from pathlib import Path
from typing import Tuple, List

from pydantic import BaseModel

from core.model import logger
from core.model.storage_helpers.SupplierFileInfo import Supplier, SupplierFileInfo

"""
sample

{
'name': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k/Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm', 
'size': 2391955766, 
'type': 'file', 
'blob_id': '2e96f326e6c2c147b1709405b4bd7ee47a6ac94d', 
'lfs': BlobLfsInfo(size=2391955766, sha256='27ae60300386eb5e825976dd3346bd6329e0effb7948961e10c90aefc003874c', pointer_size=135), 
'last_commit': None, 
'security': None
}
"""


class BlobLfsInfo(BaseModel):
    size: int
    sha256: str
    pointer_size: int


class HfFileInfo(SupplierFileInfo):
    name: str
    size: int
    type: str
    blob_id: str
    lfs: BlobLfsInfo
    last_commit: None
    security: None

    @property
    def supplier(self):
        return Supplier.HUGGINGFACE

    @staticmethod
    def model_data(split_name: List[str], model_name: str | None = None) -> Tuple[str, str, str]:
        model_name = split_name[1] if model_name is None else model_name
        file = split_name[2]
        repo = "/".join(split_name).replace(f"/{file}", "")
        return model_name, file, repo

    @property
    def huggingface_model_info_path(self):
        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper
        from core.model.ModelPath import ModelPath
        model_name, file, repo = HfFileInfo.model_data(self.name.split("/"))
        return RkllamaStorageHelper.huggingface_model_info_path_using_model_dir(
            model_dir=ModelPath.model_dir_using_model_name(model_name)
        )

    @property
    def size(self):
        return self.lfs.size

    @property
    def lfs_sha256(self):
        return self.lfs.sha256

    @staticmethod
    def last_commit_to_last_modified(last_commit):
        raise Exception("not implemented")

    @classmethod
    def load(cls, file_path: str | Path):
        logger.debug(f"HfFileInfo.load(file_path={file_path})")
        with open(file_path, "r") as f:
            return HfFileInfo(**json.load(f))

    def save(self, file_path: str| Path):
        logger.debug(f"HfFileInfo.save(file_path={file_path})")
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2, by_alias=True))