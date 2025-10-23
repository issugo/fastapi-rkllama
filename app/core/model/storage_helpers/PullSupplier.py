from enum import Enum
from logging import Logger
from typing import AsyncGenerator, Any, Tuple

from core.model.ModelFile import ModelFileInfo, ModelFile
from core.model.ModelType import ModelType


class Supplier(str, Enum):
    HUGGINGFACE = "HUGGINGFACE"
    OLLAMA = "OLLAMA"

    def is_ollama(self):
        return self is Supplier.OLLAMA

    def is_huggingface(self):
        return self is Supplier.HUGGINGFACE


class PullSupplier:

    @property
    def logger(self) -> Logger:
        raise Exception("abstract method")

    def error(self, message: str, exception = None) -> Any:
        if exception:
            self.logger.error(f"Error: {message}", exc_info=exception)
        return f"Error: {message}\n"

    def check_params(self) -> Any | None:
        raise Exception("abstract method")

    def model_data(self) -> Tuple[str, str, str|None, Supplier]:
        raise Exception("abstract method")

    def model_type(self, model_name, file, repo) -> Tuple[ModelType | None, Any]:
        raise Exception("abstract method")

    def file_info(self, model_name, file, repo, model_type, supplier: Supplier) -> Tuple[Any | None, str | None, ModelType | None, Any]:
        raise Exception("abstract method")

    def check_file_info(self, model_name, file, repo, model_type, file_info) -> Tuple[Any | None, Any]:
        raise Exception("abstract method")

    def size_and_digest(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any , Any]:
        raise Exception("abstract method")

    def model_file_info(self, model_name, file, repo, model_type, file_info, supplier: Supplier) -> Tuple[Any | None, Any | None, str | None, ModelType | None, Any]:
        raise Exception("abstract method")

    def check_model_file_info(self, model_name, file, repo, model_type, file_info, model_file_info) -> Tuple[Any | None, Any]:
        raise Exception("abstract method")

    def create_generic_model_file_info(self, file: str, model_name: str, model_type: ModelType | None, repo: str,
                                       supplier: Supplier,
                                       total_size: int,
                                       file_info,
                                       model_file_info) -> ModelFileInfo:
        raise Exception("abstract method")

    def create_generic_model_file(self, generic_model_file_info: ModelFileInfo,
                                  file_info, model_file_info,
                                  model_type: ModelType | None, repo: str) -> Tuple[ModelFile, ModelFileInfo]:
        raise Exception("abstract method")

    def lock_model(self, model_file) ->  Tuple[int | None, Any]:
        raise Exception("abstract method")

    def model_download_url(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any]:
        raise Exception("abstract method")

    @property
    def content_type(self) -> str:
        raise Exception("abstract method")

    def format_progress(self, digest: str, progress: int, total: int, completed: int) -> Any:
        raise Exception("abstract method")

    def format_success(self, digest: str) -> Any:
        raise Exception("abstract method")

    def format_error(self, error: str) -> Any:
        raise Exception("abstract method")
