import os
from pathlib import Path
from typing import Any, Tuple

from core.api.parameters import OllamaPullResponse
from core.config import config_utils
from core.config.config_utils import get_settings
from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelInfo import OllamaModelInfo
from core.model.ModelMetadata import SimpleModelMetadata
from core.model.ModelType import ModelType
from core.model.OllamaManifest import OllamaManifest, VND_OLLAMA_IMAGE_MODEL, VND_OLLAMA_IMAGE_SYSTEM, \
    OllamaManifestModelLayer
from core.model.storage_helpers.OllamaFileSystem import OllamaFileSystem
from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper
from core.model.storage_helpers.PullSupplier import PullSupplier, Supplier


DEFAULT_CONTENT_TYPE = "application/x-ndjson"

class OllamaPullSupplier(PullSupplier):

    def model_type(self, model_name, file, repo) -> Tuple[ModelType | None, Any]:
        # cannot compute the model type before getting the file info
        return None, None

    def file_info(self, model_name, file, repo, model_type, supplier: Supplier) -> Tuple[
        Any | None, str | None, ModelType | None, Any]:
        if supplier != Supplier.OLLAMA:
            return None, None, None, "Error: Invalid supplier.\n"

        # when use Hugging Face HfFileSystem to get the file metadata, have:
        # model_name=Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # file=Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm,
        # repo=dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # supplier=Supplier.HUGGINGFACE

        ollama_model_manifest_path = \
            OllamaStorageHelper.ollama_model_manifest_path(
                model_name=model_name,
                tag=file,
            )
        ollama_manifest: OllamaManifest | None = None

        self.logger.debug(f"ollama_model_manifest_path={ollama_model_manifest_path}")
        if os.path.exists(ollama_model_manifest_path):
            try:
                ollama_manifest = OllamaManifest.load(ollama_manifest_path=Path(ollama_model_manifest_path))
            except Exception as e:
                self.logger.error(f"Error reading model ollama manifest: {str(e)}")

        try:
            if ollama_manifest is None:
                fs = OllamaFileSystem()
                manifest = fs.manifest(model_name=model_name, target_tag=file)
                self.logger.debug(f"manifest={manifest}")

                ollama_manifest = OllamaManifest(**manifest)
            self.logger.debug(f"ollama_manifest={ollama_manifest}")

            """
            ollama_manifest: 
            schemaVersion = 2
            mediaType = 'application/vnd.docker.distribution.manifest.v2+json'
            config = OllamaManifestConfig(mediaType='application/vnd.docker.container.image.v1+json',
                                          digest='sha256:161ddde4c9cd07c9f1ccb4e0167c434bce72caeb3fc1844262fa66bc877b0426',
                                          size=487)
            layers = [OllamaManifestLayer(mediaType='application/vnd.ollama.image.model',
                                          digest='sha256:5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6',
                                          size=1929903008, from_=None),
                      OllamaManifestLayer(mediaType='application/vnd.ollama.image.system',
                                          digest='sha256:66b9ea09bd5b7099cbb4fc820f31b575c0366fa439b08245566692c6784e281e',
                                          size=68, from_=None),
                      OllamaManifestLayer(mediaType='application/vnd.ollama.image.template',
                                          digest='sha256:eb4402837c7829a690fa845de4d7f3fd842c2adee476d5341da8a46ea9255175',
                                          size=1482, from_=None),
                      OllamaManifestLayer(mediaType='application/vnd.ollama.image.license',
                                          digest='sha256:b5c0e5cf74cf51af1ecbc4af597cfcd13fd9925611838884a681070838a14a50',
                                          size=7387, from_=None)]
            """

            return ollama_manifest, OllamaFileSystem.model_path(model_name), model_type, None
        except Exception as e:
            return None, None, None, f"Error: {str(e)}\n"

    def check_file_info(self, model_name, file, repo, model_type, file_info) -> Tuple[Any | None, Any]:
        if not file_info:
            return None, "Error: missing manifest.\n"

        # test that manifest contains at least model and system layers
        if not file_info.config:
            return None, "Error: missing config in manifest.\n"
        if not file_info.layers:
            return None, "Error: missing layers in manifest.\n"
        if len(file_info.layers) < 2:
            return None, "Error: missing layers in manifest.\n"
        model_manifest_layers = [layer.mediaType for layer in file_info.layers]
        if not VND_OLLAMA_IMAGE_MODEL in model_manifest_layers:
            return None, "Error: missing model layer in manifest.\n"
        if not VND_OLLAMA_IMAGE_SYSTEM in model_manifest_layers:
            return None, "Error: missing system layer in manifest.\n"

        return file_info, None

    def size_and_digest(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any, Any]:
        total_size = file_info.size  # File size in bytes
        if total_size == 0:
            return None, None, "Error: Unable to retrieve file size.\n"

        digest = file_info.lfs_sha256
        if not digest:
            return None, None, "Error: Unable to retrieve file digest.\n"

        return total_size, digest, None

    def model_file_info(self, model_name, file, repo, model_type, file_info, supplier: Supplier) -> Tuple[
        Any | None, Any | None, str | None, ModelType | None, Any]:
        if supplier != Supplier.OLLAMA:
            return None, None, None, None, "Error: Invalid supplier.\n"

        # when use Hugging Face HfFileSystem to get the file metadata, have:
        # model_name=Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # file=Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm,
        # repo=dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # supplier=Supplier.HUGGINGFACE

        fs = OllamaFileSystem()

        ollama_model_config_path = \
            file_info.ollama_model_config_path
        ollama_model_info: OllamaModelInfo | None = None

        self.logger.debug(f"ollama_model_config_path={ollama_model_config_path}")
        if os.path.exists(ollama_model_config_path):
            try:
                ollama_model_info = OllamaModelInfo.load(file_path=ollama_model_config_path)
            except Exception as e:
                self.logger.error(f"Error reading model ollama manifest: {str(e)}")

        try:
            if ollama_model_info is None:
                config_digest = file_info.config.digest
                ollama_config, info = fs.load_config(config_digest=config_digest, model_name=model_name,
                                                     target_tag=file)
                self.logger.debug(f"ollama_config={ollama_config}, info={info}")

                """
                ollama_config: {'model_format': 'gguf', 'model_family': 'qwen2', 'model_families': ['qwen2'],
                                 'model_type': '1.5B', 'file_type': 'Q4_K_M', 'architecture': 'amd64', 'os': 'linux',
                                 'rootfs': {'type': 'layers', 'diff_ids': [
                                     'sha256:183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4',
                                     'sha256:75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de',
                                     'sha256:9bebd78bf5bc92d41d5f3aab3ee66c891376b4eb4cf433edc2533c2f5f9c95a6',
                                     'sha256:832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e']}}
                """
                ollama_model_info = OllamaModelInfo(**ollama_config)
            self.logger.debug(f"ollama_model_info={ollama_model_info}")

            for _model_type in ModelType:
                if _model_type.value.upper() == ollama_model_info.model_format.upper():
                    model_type = _model_type
                    break
            self.logger.debug(f"model_type={model_type}")

            ollama_model_info.ollama_manifest = file_info

            return ollama_model_info, ollama_model_info.ollama_manifest, OllamaFileSystem.model_path(
                model_name), model_type, None
        except Exception as e:
            self.logger.exception(f"Error reading model ollama manifest: {str(e)}", exc_info=e)
            return None, None, None, None, f"Error: {str(e)}\n"

    def check_model_file_info(self, model_name, file, repo, model_type, file_info, model_file_info) -> Tuple[
        Any | None, Any]:
        if not model_file_info.ollama_manifest:
            return None, "Error: missing manifest.\n"

        # test that manifest contains at least model and system layers
        _, error = self.check_file_info(
            model_name=model_name, file=file, repo=repo, model_type=model_type,
            file_info=model_file_info.ollama_manifest)
        if error:
            return None, error

        # TODO: check that model_file_info is valid

        return model_file_info, None

    def create_generic_model_file_info(self, file: str, model_name: str, model_type: ModelType | None, repo: str,
                                       supplier: Supplier,
                                       total_size: int,
                                       file_info: OllamaManifest,
                                       model_file_info: OllamaModelInfo) -> ModelFileInfo:
        generic_model_file_info: ModelFileInfo = ModelFileInfo(
            model_name=model_name,
            model_type=model_type,
            endpoint_model_file=file,
            endpoint_model_file_size=total_size,
            huggingface_path=None,
            ollama_path=repo,
        )
        generic_model_file_info.ollama_file_info = file_info
        generic_model_file_info.ollama_model_info = model_file_info
        self.logger.debug(f"generic_model_file_info={generic_model_file_info.model_dump_json()}")
        self.logger.debug(f"ollama_data={generic_model_file_info.ollama_model_info}")

        return generic_model_file_info

    def create_generic_model_file(self, generic_model_file_info: ModelFileInfo,
                                  file_info, model_file_info,
                                  model_type, repo) -> Tuple[ModelFile, ModelFileInfo]:
        generic_model_file_info_dump = generic_model_file_info.model_dump()
        self.logger.debug(f"generic_model_file_info_dump={generic_model_file_info_dump}")
        generic_model_file: ModelFile = ModelFile.create(
            model_file_info=generic_model_file_info,
            default_model_config=get_settings().model)
        self.logger.debug(f"generic_model_file={generic_model_file.model_dump_json()}")
        self.logger.debug(
            f"generic_model_file.simple_model_metadata={generic_model_file.simple_model_metadata.model_dump_json()}")
        generic_model_file.ollama_model_info = model_file_info
        generic_model_file.ollama_file_info = file_info
        generic_model_file_info = ModelFileInfo(**generic_model_file_info_dump)
        if isinstance(generic_model_file.model_metadata, SimpleModelMetadata):
            generic_model_file_info.simple_model_metadata = generic_model_file.model_metadata
        generic_model_file_info.ollama_model_info = model_file_info
        generic_model_file_info.ollama_file_info = file_info
        return generic_model_file, generic_model_file_info

    def lock_model(self, model_file) -> Tuple[int | None, Any]:
        if model_file.is_locked():
            return None, "Error: Model is currently locked.\n"

        lock_id = model_file.lock_model()
        return lock_id, None

    def model_download_url(self, model_name, file, repo, model_type, file_info: OllamaManifest) -> Tuple[Any, Any]:
        try:
            ollama_manifest_model_layer: OllamaManifestModelLayer = file_info.ollama_manifest_model_layer
            if not ollama_manifest_model_layer:
                return None, "Error: missing model layer in manifest.\n"

            model_digest = ollama_manifest_model_layer.digest
            fs = OllamaFileSystem()

            url = fs.model_url(model_digest=model_digest, model_name=model_name)
            return url, None
        except Exception as e:
            err_msg = f"Error building ollama model download url: {str(e)}"
            self.logger.exception(err_msg, exc_info=e)
            return None, f"{err_msg}\n"

    @property
    def content_type(self) -> str:
        return DEFAULT_CONTENT_TYPE

    def format_progress(self, digest: str, progress: int, total: int, completed: int) -> Any:
        return OllamaPullResponse(
                status="downloading model",
                digest=f"sha256:{digest}",
                total=total,
                completed=completed,
            ).model_dump_json().encode() + b"\n"

    def format_success(self, digest: str) -> Any:
        return OllamaPullResponse(
                status="success",
                digest=f"sha256:{digest}",
            ).model_dump_json().encode() + b"\n"

    def format_error(self, error: str) -> Any:
        return OllamaPullResponse(
                status=f"{error}",
            ).model_dump_json().encode() + b"\n"

