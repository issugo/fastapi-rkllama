from pydantic import BaseModel

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


class HfFileInfo(BaseModel):
    name: str
    size: int
    type: str
    blob_id: str
    lfs: BlobLfsInfo
    last_commit: None
    security: None
