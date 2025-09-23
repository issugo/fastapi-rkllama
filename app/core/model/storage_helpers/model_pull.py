import os
from logging import Logger
from typing import AsyncGenerator, Any, Tuple

import requests
from starlette.requests import Request
from starlette.responses import StreamingResponse

from core.config import config_utils
from core.model.ModelFile import ModelFile, ModelFileInfo
from core.model.ModelType import ModelType
from core.model.storage_helpers.OllamaModelStorageHelper import OllamaModelStorageHelper
from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper
from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper


class PullSupplier:

    @property
    def logger(self) -> Logger:
        pass

    def error(self, message: str) -> Any:
        pass

    def check_params(self) -> Any | None:
        pass

    def model_data(self) -> Tuple[str, str, str]:
        pass

    def model_type(self, model_name, file, repo) -> Tuple[ModelType | None, Any]:
        pass

    def file_info(self, model_name, file, repo, model_type) -> Tuple[Any | None, Any]:
        pass

    def check_file_info(self, model_name, file, repo, model_type, file_info) -> Tuple[Any | None, Any]:
        pass

    def size_and_digest(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any , Any]:
        pass

    def lock_model(self, model_file) ->  Tuple[int | None, Any]:
        pass

    def model_download_url(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any]:
        pass

    @property
    def content_type(self) -> str:
        pass

    def format_progress(self, progress: int) -> Any:
        pass

def pull_model(request: Request,
               pull_supplier: PullSupplier,
               ):

    # Use the appropriate content type for streaming responses
    ## is_ollama_request = request.path.startswith('/api/')
    pull_supplier.logger.debug(f"request.url={request.url}")

    check_result = pull_supplier.check_params()
    if check_result is not None:
        pull_supplier.logger.error(check_result)
        return check_result

    model_name, file, repo = pull_supplier.model_data()
    pull_supplier.logger.debug(f"model_name={model_name}, file={file}, repo={repo}")

    model_type, error = pull_supplier.model_type(model_name, file, repo)
    if error is not None:
        pull_supplier.logger.error(error)
        return error
    if model_type is None:
        error = pull_supplier.error(f"Invalid model type '{model_type}'\n")
        pull_supplier.logger.error(error)
        return error

    try:
        hf_file_info, error = pull_supplier.file_info(model_name, file, repo, model_type)
        if error is not None:
            pull_supplier.logger.error(error)
            return error

        hf_file_info, error = pull_supplier.check_file_info(model_name, file, repo, model_type, hf_file_info)
        if error is not None:
            pull_supplier.logger.error(error)
            return error
        if hf_file_info is None:
            error = pull_supplier.error(f"Invalid file info'\n")
            pull_supplier.logger.error(error)
            return error

        total_size, digest, error = pull_supplier.size_and_digest(model_name, file, repo, model_type, hf_file_info)
        if error is not None:
            pull_supplier.logger.error(error)
            return error

        # Create the configuration file for model
        if model_type is None:
            model_type = ModelType.get_model_type_from_endpoint_model_file(file)

        pull_supplier.logger.debug(f"{model_type}")

        model_file, ollama_model_storage_helper, ollama_storage_helper, rkllama_storage_helper = pull_model_build_data(
            digest, file, model_name, model_type, pull_supplier, repo, total_size)

        lock_id, error = pull_supplier.lock_model(model_file)
        if error is not None:
            pull_supplier.logger.error(error)
            return error
        elif lock_id > 0:

            model_blob_path = ollama_model_storage_helper.model_blob_path

            try:
                # Download the file with progress
                url, error = pull_supplier.model_download_url(model_name, file, repo, model_type, hf_file_info)
                if error is not None:
                    pull_supplier.logger.error(error)
                    return error
                else:
                    with (
                        requests.get(url, stream=True) as r,
                        open(model_blob_path, "wb") as f,
                    ):
                        downloaded_size = 0
                        chunk_size = 8192  # 8KB

                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                progress = int((downloaded_size / total_size) * 100)
                                pull_supplier.logger.debug(f"{progress}%\n")

                    rkllama_storage_helper.store()
                    ollama_storage_helper.store()

                    model_file.unlock_model(lock_id)

                    # success
                    return None

            except Exception as download_error:
                error = pull_model_clean(download_error, lock_id, model_blob_path, model_file,
                                         ollama_model_storage_helper, ollama_storage_helper, pull_supplier,
                                         rkllama_storage_helper)
                return error

    except Exception as e:
        error = pull_supplier.error(f"Error: {str(e)}\n")
        pull_supplier.logger.error(error)
        return error


def pull_model_build_data(digest, file: str, model_name: str, model_type: ModelType | None, pull_supplier: PullSupplier,
                          repo: str, total_size) -> tuple[
    ModelFile, OllamaModelStorageHelper, OllamaStorageHelper, RkllamaStorageHelper]:
    model_file_info: ModelFileInfo = ModelFileInfo(
        model_name=model_name,
        model_type=model_type,
        huggingface_path=repo,
        endpoint_model_file=file,
        endpoint_model_file_size=total_size,
    )
    pull_supplier.logger.debug(f"model_file_info={model_file_info.model_dump_json()}")
    pull_supplier.logger.debug(f"hf_data={model_file_info.huggingface_model_info}")

    ollama_model_storage_helper: OllamaModelStorageHelper = OllamaModelStorageHelper(
        model_path=model_file_info,
        sha256_digest=digest
    )
    model_file: ModelFile = ModelFile.create(
        model_file_info=model_file_info,
        default_model_config=config_utils.rkllama_config.model)
    pull_supplier.logger.debug(f"model_file={model_file.model_dump_json()}")
    pull_supplier.logger.debug(f"model_file.simple_model_metadata={model_file.simple_model_metadata.model_dump_json()}")

    rkllama_storage_helper: RkllamaStorageHelper = RkllamaStorageHelper(
        ollama_model_storage_helper=ollama_model_storage_helper,
        model_file=model_file
    )

    ollama_storage_helper: OllamaStorageHelper = OllamaStorageHelper(
        ollama_model_storage_helper=ollama_model_storage_helper,
        model_file=model_file
    )
    return model_file, ollama_model_storage_helper, ollama_storage_helper, rkllama_storage_helper


def pull_model_clean(download_error: Exception, lock_id: int | None, model_blob_path: str | Any,
                     model_file: ModelFile, ollama_model_storage_helper: OllamaModelStorageHelper,
                     ollama_storage_helper: OllamaStorageHelper, pull_supplier: PullSupplier,
                     rkllama_storage_helper: RkllamaStorageHelper) -> Any:
    # Remove the file if an error occurs during download
    ollama_storage_helper.clean()
    rkllama_storage_helper.clean()
    ollama_model_storage_helper.clean()
    if os.path.exists(model_blob_path):
        os.remove(model_blob_path)
    model_file.unlock_model(lock_id)
    error = pull_supplier.error(f"Error during download: {str(download_error)}\n")
    pull_supplier.logger.error(error)
    return error


def pull_model_stream(request: Request,
                      pull_supplier: PullSupplier,
                      ):

    # Use the appropriate content type for streaming responses
    ## is_ollama_request = request.path.startswith('/api/')
    pull_supplier.logger.debug(f"request.url={request.url}")

    async def generate_progress() -> AsyncGenerator[str, None]:
        check_result = pull_supplier.check_params()
        if check_result is not None:
            pull_supplier.logger.error(check_result)
            yield check_result

        model_name, file, repo = pull_supplier.model_data()
        pull_supplier.logger.debug(f"model_name={model_name}, file={file}, repo={repo}")

        model_type, error = pull_supplier.model_type(model_name, file, repo)
        if error is not None:
            pull_supplier.logger.error(error)
            yield error
        if model_type is None:
            error = pull_supplier.error(f"Invalid model type '{model_type}'\n")
            pull_supplier.logger.error(error)
            yield error

        try:
            file_info, error = pull_supplier.file_info(model_name, file, repo, model_type)
            if error is not None:
                pull_supplier.logger.error(error)
                yield error

            file_info, error = pull_supplier.check_file_info(model_name, file, repo, model_type, file_info)
            if error is not None:
                pull_supplier.logger.error(error)
                yield error
            if file_info is None:
                error = pull_supplier.error(f"Invalid file info'\n")
                pull_supplier.logger.error(error)
                yield error

            total_size, digest, error = pull_supplier.size_and_digest(model_name, file, repo, model_type, file_info)
            if error is not None:
                pull_supplier.logger.error(error)
                yield error

            # Create the configuration file for model
            if model_type is None:
                model_type = ModelType.get_model_type_from_endpoint_model_file(file)

            pull_supplier.logger.debug(f"{model_type}")

            model_file, ollama_model_storage_helper, ollama_storage_helper, rkllama_storage_helper = pull_model_build_data(
                digest, file, model_name, model_type, pull_supplier, repo, total_size)

            lock_id, error = pull_supplier.lock_model(model_file)
            if error is not None:
                pull_supplier.logger.error(error)
                yield error
            elif lock_id > 0:

                model_blob_path = ollama_model_storage_helper.model_blob_path

                try:
                    # Download the file with progress
                    url, error = pull_supplier.model_download_url(model_name, file, repo, model_type, file_info)
                    if error is not None:
                        pull_supplier.logger.error(error)
                        yield error
                    else:
                        with (
                            requests.get(url, stream=True) as r,
                            open(model_blob_path, "wb") as f,
                        ):
                            downloaded_size = 0
                            chunk_size = 8192  # 8KB

                            for chunk in r.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    progress = int((downloaded_size / total_size) * 100)
                                    pull_supplier.logger.debug(f"{progress}%\n")
                                    yield pull_supplier.format_progress(progress)

                        rkllama_storage_helper.store()
                        ollama_storage_helper.store()

                        model_file.unlock_model(lock_id)

                except Exception as download_error:
                    error = pull_model_clean(download_error, lock_id, model_blob_path, model_file,
                                             ollama_model_storage_helper, ollama_storage_helper, pull_supplier,
                                             rkllama_storage_helper)
                    yield error
                    return

        except Exception as e:
            error = pull_supplier.error(f"Error: {str(e)}\n")
            pull_supplier.logger.error(error)
            yield error

    return StreamingResponse(
        generate_progress(), headers={"Content-Type": pull_supplier.content_type}
    )
