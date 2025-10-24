import os
from pathlib import Path
from typing import AsyncGenerator, Any

import requests
from starlette.requests import Request
from starlette.responses import StreamingResponse

from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelType import ModelType
from core.model.storage_helpers.OllamaModelStorageHelper import OllamaModelStorageHelper
from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper
from core.model.storage_helpers.PullSupplier import PullSupplier, Supplier
from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper


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
        # when the supplier is Ollama, file_info is an OllamaManifest
        # when the supplier is Rkllama, file_info is a HfFileInfo
        file_info, repo, model_type, error = pull_supplier.file_info(
            model_name=model_name, file=file, repo=repo, model_type=model_type, supplier=supplier)
        if error is not None:
            pull_supplier.logger.error(error)
            return error

        file_info, error = pull_supplier.check_file_info(model_name, file, repo, model_type, file_info)
        if error is not None:
            pull_supplier.logger.error(error)
            return error
        if file_info is None:
            error = pull_supplier.error(f"Invalid file info'\n")
            pull_supplier.logger.error(error)
            return error

        total_size, digest, error = pull_supplier.size_and_digest(model_name, file, repo, model_type, file_info)
        if error is not None:
            pull_supplier.logger.error(error)
            return error

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
            return error

        if model_type is None:
            error = pull_supplier.error(f"Invalid model type '{model_type}'\n")
            pull_supplier.logger.error(error)
            return error

        model_file_info, error = pull_supplier.check_model_file_info(
            model_name=model_name, file=file, repo=repo, model_type=model_type,
            file_info=file_info, model_file_info=model_file_info)
        if error is not None:
            pull_supplier.logger.error(error)
            return error
        if model_file_info is None:
            error = pull_supplier.error(f"Invalid model file info'\n")
            pull_supplier.logger.error(error)
            return error

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

        generic_model_file, generic_model_file_info = pull_supplier.create_generic_model_file(
            generic_model_file_info=generic_model_file_info,
            file_info=file_info,
            model_file_info=model_file_info,
            model_type=model_type,
            repo=repo,
        )
        pull_supplier.logger.debug(
            f"pull_model_stream(): generic_model_file={generic_model_file}")

        # debug
        if supplier.is_huggingface():
            huggingface_model_info_path = RkllamaStorageHelper.huggingface_model_info_path(generic_model_file_info)
            pull_supplier.logger.debug(
                f"pull_model_stream(): huggingface_model_info_path={huggingface_model_info_path}")
        elif supplier.is_ollama():
            ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(model_path=generic_model_file_info,
                                                                                ollama_manifest=file_info)
            pull_supplier.logger.debug(f"pull_model_stream(): ollama_model_info_path={ollama_model_info_path}")


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
            file_info, repo, model_type, error = pull_supplier.file_info(
                model_name=model_name, file=file, repo=repo, model_type=model_type, supplier=supplier)
            if error is not None:
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)

            file_info, error = pull_supplier.check_file_info(model_name, file, repo, model_type, file_info)
            if error is not None:
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)
            if file_info is None:
                error = pull_supplier.error(f"Invalid file info'\n")
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)

            total_size, digest, error = pull_supplier.size_and_digest(model_name, file, repo, model_type, file_info)
            if error is not None:
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)

            # Create the configuration file for a model
            if model_type is None:
                model_type = ModelType.get_model_type_from_endpoint_model_file(file)

            pull_supplier.logger.debug(f"model_type={model_type}")

            model_file_info, file_info, repo, model_type, error = pull_supplier.model_file_info(
                model_name=model_name, file=file, repo=repo, model_type=model_type,
                file_info=file_info, supplier=supplier)
            if error is not None:
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)

            if model_type is None:
                error = pull_supplier.error(f"Invalid model type '{model_type}'\n")
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)

            model_file_info, error = pull_supplier.check_model_file_info(
                model_name=model_name, file=file, repo=repo, model_type=model_type,
                file_info=file_info, model_file_info=model_file_info)
            if error is not None:
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)
            if model_file_info is None:
                error = pull_supplier.error(f"Invalid model file info'\n")
                pull_supplier.logger.error(error)
                yield pull_supplier.format_error(error)

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

            generic_model_file, generic_model_file_info = pull_supplier.create_generic_model_file(
                generic_model_file_info=generic_model_file_info,
                file_info=file_info,
                model_file_info=model_file_info,
                model_type=model_type,
                repo=repo,
            )
            pull_supplier.logger.debug(
                f"pull_model_stream(): generic_model_file={generic_model_file}")

            # debug
            if supplier.is_huggingface():
                huggingface_model_info_path = RkllamaStorageHelper.huggingface_model_info_path(generic_model_file_info)
                pull_supplier.logger.debug(f"pull_model_stream(): huggingface_model_info_path={huggingface_model_info_path}")
            elif supplier.is_ollama():
                ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(model_path=generic_model_file_info,
                                                                                    ollama_manifest=file_info)
                pull_supplier.logger.debug(f"pull_model_stream(): ollama_model_info_path={ollama_model_info_path}")

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
