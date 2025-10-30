import os
from pathlib import Path
from typing import AsyncGenerator, Any

import requests
from starlette.requests import Request
from starlette.responses import StreamingResponse

from core.model.Model import Model
from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelInfo import ModelInfo
from core.model.ModelType import ModelType
from core.model.storage_helpers.OllamaModelStorageHelper import OllamaModelStorageHelper
from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper
from core.model.storage_helpers.PullSupplier import PullSupplier
from core.model.storage_helpers.SupplierFileInfo import Supplier
from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper

def _create_specific(model_name: str, file: str, repo: str, model_type: ModelType,
                     supplier: Supplier, pull_supplier: PullSupplier) -> tuple[Any, Any, Any, Any, str, str, str|None, ModelType | None, str|None]:
    """
    return: model_file_info, file_info, total_size, digest, model_name, file, repo, model_type, error
    """
    # when the supplier is Ollama, file_info is an OllamaManifest
    # when the supplier is Rkllama, file_info is a HfFileInfo
    file_info, repo, model_type, error = pull_supplier.file_info(
        model_name=model_name, file=file, repo=repo, model_type=model_type, supplier=supplier)
    if error is not None:
        pull_supplier.logger.error(error)
        return None, None, None, None, model_name, file, repo, model_type, error

    file_info, error = pull_supplier.check_file_info(model_name, file, repo, model_type, file_info)
    if error is not None:
        pull_supplier.logger.error(error)
        return None, None, None, None, model_name, file, repo, model_type, error
    if file_info is None:
        error = pull_supplier.error(f"Invalid file info'\n")
        pull_supplier.logger.error(error)
        return None, None, None, None, model_name, file, repo, model_type, error

    pull_supplier.logger.debug(f"file_info.cls={file_info.__class__}")

    total_size = file_info.size  # File size in bytes
    if total_size == 0:
        error = pull_supplier.error(f"Unable to retrieve file size.\n")
        pull_supplier.logger.error(error)
        return None, file_info, None, None, model_name, file, repo, model_type, error

    digest = file_info.digest
    if not digest:
        error = pull_supplier.error(f"Unable to retrieve file digest.\n")
        pull_supplier.logger.error(error)
        return None, file_info, total_size, None, model_name, file, repo, model_type, error

    # Create the configuration file for the model
    if model_type is None:
        model_type = ModelType.get_model_type_from_endpoint_model_file(file)

    pull_supplier.logger.debug(f"model_type={model_type}")

    # when the supplier is Ollama, file_info is an OllamaModelInfo
    # when the supplier is Rkllama, file_info is a HFModelInfo
    model_file_info, file_info, repo, model_type, error = pull_supplier.model_file_info(
        model_name=model_name, file=file, repo=repo, model_type=model_type,
        file_info=file_info, supplier=supplier)
    if error is not None:
        pull_supplier.logger.error(error)
        return model_file_info, file_info, total_size, digest, model_name, file, repo, model_type, error

    if model_type is None:
        error = pull_supplier.error(f"Invalid model type '{model_type}'\n")
        pull_supplier.logger.error(error)
        return model_file_info, file_info, total_size, digest, model_name, file, repo, None, error

    model_file_info, error = pull_supplier.check_model_file_info(
        model_name=model_name, file=file, repo=repo, model_type=model_type,
        file_info=file_info, model_file_info=model_file_info)
    if error is not None:
        pull_supplier.logger.error(error)
        return model_file_info, file_info, total_size, digest, model_name, file, repo, model_type, error
    if model_file_info is None:
        error = pull_supplier.error(f"Invalid model file info'\n")
        pull_supplier.logger.error(error)
        return None, file_info, total_size, digest, model_name, file, repo, model_type, error

    return model_file_info, file_info, total_size, digest, model_name, file, repo, model_type, None


def _create_generic(model_name: str, file: str, repo: str, model_type: ModelType,
                model_file_info: Any, file_info: Any, total_size: Any, digest: Any,
                supplier: Supplier, pull_supplier: PullSupplier)  -> tuple[ModelFile | None, ModelFileInfo| None, Model | None, ModelInfo| None, str|None]:
    """
    returns: generic_model_file, generic_model_file_info, generic_model, generic_model_info, error
    """
    generic_model_info = pull_supplier.create_generic_model_info(
        file=file,
        model_name=model_name,
        model_type=model_type,
        repo=repo,
        supplier=supplier,
        total_size=total_size,
        digest=digest,
        file_info=file_info,
        model_file_info=model_file_info,
    )
    pull_supplier.logger.debug(
        f"pull_model_stream(): generic_model_info={generic_model_info}")
    if generic_model_info is None:
        error = pull_supplier.error(f"Invalid Model info'\n")
        pull_supplier.logger.error(error)
        return None, None, None, None, error

    generic_model, generic_model_info = pull_supplier.create_generic_model(
        generic_model_info=generic_model_info,
        file_info=file_info,
        model_file_info=model_file_info,
        model_type=model_type,
        repo=repo,
    )
    pull_supplier.logger.debug(
        f"pull_model_stream(): generic_model={generic_model}")
    if generic_model is None:
        error = pull_supplier.error(f"Invalid Model'\n")
        pull_supplier.logger.error(error)
        return None, None, None, generic_model_info, error

    generic_model_file_info = pull_supplier.create_generic_model_file_info(
        file=file,
        model_name=model_name,
        model_type=model_type,
        repo=repo,
        supplier=supplier,
        total_size=total_size,
        file_info=file_info,
        model_file_info=model_file_info,
    )
    pull_supplier.logger.debug(
        f"pull_model_stream(): generic_model_file_info={generic_model_file_info}")
    if generic_model_file_info is None:
        error = pull_supplier.error(f"Invalid Modelfile info'\n")
        pull_supplier.logger.error(error)
        return None, None, generic_model, generic_model_info, error

    generic_model_file, generic_model_file_info = pull_supplier.create_generic_model_file(
        generic_model_file_info=generic_model_file_info,
        file_info=file_info,
        model_file_info=model_file_info,
        model=generic_model,
        model_type=model_type,
        repo=repo,
    )
    pull_supplier.logger.debug(
        f"pull_model_stream(): generic_model_file={generic_model_file}")
    if generic_model_file is None:
        error = pull_supplier.error(f"Invalid Modelfile'\n")
        pull_supplier.logger.error(error)
        return None, generic_model_file_info, generic_model, generic_model_info, error

    # debug
    if supplier.is_huggingface():
        huggingface_model_info_path = RkllamaStorageHelper.huggingface_model_info_path(generic_model_file_info)
        pull_supplier.logger.debug(
            f"pull_model_stream(): huggingface_model_info_path={huggingface_model_info_path}")
    elif supplier.is_ollama():
        ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(model_path=generic_model_file_info,
                                                                            ollama_manifest=file_info)
        pull_supplier.logger.debug(f"pull_model_stream(): ollama_model_info_path={ollama_model_info_path}")

    return generic_model_file, generic_model_file_info, generic_model, generic_model_info, None


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

    model_name, file, repo, supplier = pull_supplier.model_data()
    pull_supplier.logger.debug(f"model_name={model_name}, file={file}, repo={repo}, supplier={supplier}")

    model_type, error = pull_supplier.model_type(model_name, file, repo)
    if error is not None:
        pull_supplier.logger.error(error)
        return error

    try:

        model_file_info, file_info, total_size, digest, model_name, file, repo, model_type, error = (
            _create_specific(model_name=model_name, file=file, repo=repo, model_type=model_type,
                             supplier=supplier, pull_supplier=pull_supplier)
        )
        if error is not None:
            return error

        generic_model_file, generic_model_file_info, generic_model, generic_model_info, error = (
            _create_generic(model_name=model_name, file=file, repo=repo, model_type=model_type,
                            model_file_info=model_file_info, file_info=file_info, total_size=total_size, digest=digest,
                            supplier=supplier, pull_supplier=pull_supplier)
        )
        if error is not None:
            return error

        ollama_model_storage_helper, ollama_storage_helper, rkllama_storage_helper = create_storage_helpers(
            generic_model_file=generic_model_file,
            generic_model_file_info=generic_model_file_info,
            file_info=file_info,
            digest=digest,
            supplier=supplier,
            pull_supplier=pull_supplier,
            total_size=total_size)

        lock_id, error = pull_supplier.lock_model(generic_model_file)
        if error is not None:
            pull_supplier.logger.error(error)
            return error
        elif lock_id > 0:

            model_blob_path, model_digest = ollama_model_storage_helper.model_blob_path
            pull_supplier.logger.debug(f"pull_model_stream(): model_blob_path={model_blob_path}")
            pull_supplier.logger.debug(f"pull_model_stream(): model_blob_path={Path(model_blob_path).resolve()}")

            try:
                # Download the file with progress
                url, error = pull_supplier.model_download_url(model_name, file, repo, model_type, file_info)
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
                    ollama_model_storage_helper.store()

                    generic_model_file.unlock_model(lock_id)

                    # success
                    return model_digest

            except Exception as download_error:
                pull_supplier.logger.exception(f"pull_model(): error={str(download_error)}", exc_info=True)
                error = pull_model_clean(
                    download_error=download_error,
                    lock_id=lock_id,
                    model_blob_path=model_blob_path,
                    generic_model_file=generic_model_file,
                    generic_model_file_info=generic_model_file_info,
                    ollama_model_storage_helper=ollama_model_storage_helper,
                    ollama_storage_helper=ollama_storage_helper,
                    rkllama_storage_helper=rkllama_storage_helper,
                    pull_supplier=pull_supplier,
                )
                return error

    except Exception as e:
        pull_supplier.logger.exception(f"pull_model(): error={str(e)}", exc_info=True)
        error = pull_supplier.error(f"Error: {str(e)}\n")
        pull_supplier.logger.error(error)
        return error


def create_storage_helpers(generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo, digest, supplier: Supplier, pull_supplier: PullSupplier,
                           total_size: int,
                           file_info: Any) -> tuple[
    OllamaModelStorageHelper, OllamaStorageHelper, RkllamaStorageHelper]:
    pull_supplier.logger.info(f"pull_model_build_data: digest={digest}, generic_model_file_info={generic_model_file_info}")

    try:
        ollama_model_storage_helper: OllamaModelStorageHelper = OllamaModelStorageHelper(
            model_path=generic_model_file_info,
            sha256_digest=digest,
            generic_model_file = generic_model_file,
            logger=pull_supplier.logger
        )

        rkllama_storage_helper: RkllamaStorageHelper = RkllamaStorageHelper(
            ollama_model_storage_helper=ollama_model_storage_helper,
            model_file=generic_model_file
        )

        ollama_storage_helper: OllamaStorageHelper = OllamaStorageHelper(
            ollama_model_storage_helper=ollama_model_storage_helper,
            model_file=generic_model_file
        )
        return ollama_model_storage_helper, ollama_storage_helper, rkllama_storage_helper
    except Exception as e:
        pull_supplier.logger.exception(f"Error building model data: {str(e)}", exc_info=True)
        raise e


def pull_model_clean(download_error: Exception, lock_id: int | None, model_blob_path: str | Any,
                     generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo, pull_supplier: PullSupplier,
                     ollama_model_storage_helper: OllamaModelStorageHelper,
                     ollama_storage_helper: OllamaStorageHelper,
                     rkllama_storage_helper: RkllamaStorageHelper) -> Any:
    # Remove the file if an error occurs during download
    ollama_storage_helper.clean(generic_model_file=generic_model_file, generic_model_file_info=generic_model_file_info)
    rkllama_storage_helper.clean(generic_model_file=generic_model_file, generic_model_file_info=generic_model_file_info)
    ollama_model_storage_helper.clean(generic_model_file=generic_model_file, generic_model_file_info=generic_model_file_info)
    if os.path.exists(model_blob_path):
        os.remove(model_blob_path)
    generic_model_file.unlock_model(lock_id)
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
            yield pull_supplier.format_error(check_result)

        model_name, file, repo, supplier = pull_supplier.model_data()
        pull_supplier.logger.debug(f"model_name={model_name}, file={file}, repo={repo}, supplier={supplier}")

        model_type, error = pull_supplier.model_type(model_name, file, repo)
        if error is not None:
            pull_supplier.logger.error(error)
            yield pull_supplier.format_error(error)

        try:
            model_file_info, file_info, total_size, digest, model_name, file, repo, model_type, error = (
                _create_specific(model_name=model_name, file=file, repo=repo, model_type=model_type,
                                 supplier=supplier, pull_supplier=pull_supplier)
            )
            if error is not None:
                yield pull_supplier.format_error(error)

            generic_model_file, generic_model_file_info, generic_model, generic_model_info, error = (
                _create_generic(model_name=model_name, file=file, repo=repo, model_type=model_type,
                                model_file_info=model_file_info, file_info=file_info, total_size=total_size,
                                digest=digest,
                                supplier=supplier, pull_supplier=pull_supplier)
            )
            if error is not None:
                yield pull_supplier.format_error(error)

            ollama_model_storage_helper, ollama_storage_helper, rkllama_storage_helper = create_storage_helpers(
                generic_model_file=generic_model_file,
                generic_model_file_info=generic_model_file_info,
                file_info=file_info,
                digest=digest,
                supplier=supplier,
                pull_supplier=pull_supplier,
                total_size=total_size)

            lock_id, error = pull_supplier.lock_model(generic_model_file)
            if error is not None:
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)
            elif lock_id > 0:

                model_blob_path, model_digest = ollama_model_storage_helper.model_blob_path
                pull_supplier.logger.debug(f"pull_model_stream(): model_blob_path={model_blob_path}")
                pull_supplier.logger.debug(f"pull_model_stream(): model_blob_path={Path(model_blob_path).resolve()}")

                try:
                    # Download the file with progress
                    url, error = pull_supplier.model_download_url(model_name, file, repo, model_type, file_info)
                    if error is not None:
                        pull_supplier.logger.error(error)
                        yield pull_supplier.format_error(error)
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
                                    yield pull_supplier.format_progress(
                                        digest=model_digest,
                                        progress=progress,
                                        total=total_size,
                                        completed=downloaded_size,
                                    )

                        rkllama_storage_helper.store()
                        ollama_storage_helper.store()
                        ollama_model_storage_helper.store()

                        generic_model_file.unlock_model(lock_id)

                        yield pull_supplier.format_success(digest=model_digest)

                except Exception as download_error:
                    pull_supplier.logger.exception(f"pull_model_stream(): error={str(download_error)}", exc_info=True)
                    error = pull_model_clean(
                        download_error=download_error,
                        lock_id=lock_id,
                        model_blob_path=model_blob_path,
                        generic_model_file=generic_model_file,
                        generic_model_file_info=generic_model_file_info,
                        ollama_model_storage_helper=ollama_model_storage_helper,
                        ollama_storage_helper=ollama_storage_helper,
                        rkllama_storage_helper=rkllama_storage_helper,
                        pull_supplier=pull_supplier,
                        )
                    yield pull_supplier.format_error(error)
                    return

        except Exception as e:
            pull_supplier.logger.exception(f"pull_model_stream(): error={str(e)}", exc_info=True)
            error = pull_supplier.error(f"Error: {str(e)}\n")
            pull_supplier.logger.error(error)
            yield pull_supplier.format_error(error)

    return StreamingResponse(
        generate_progress(), headers={"Content-Type": pull_supplier.content_type}
    )
