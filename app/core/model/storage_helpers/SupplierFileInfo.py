from enum import Enum

from pydantic import BaseModel


class Supplier(str, Enum):
    HUGGINGFACE = "HUGGINGFACE"
    OLLAMA = "OLLAMA"

    def is_ollama(self):
        return self is Supplier.OLLAMA

    def is_huggingface(self):
        return self is Supplier.HUGGINGFACE


class SupplierFileInfo(BaseModel):
    # @property
    def supplier(self):
        raise Exception("abstract method")

    # @property
    def size(self):
        raise Exception("abstract method")

    # @property
    def lfs_sha256(self):
        raise Exception("abstract method")

    @property
    def digest(self):
        return self.lfs_sha256
