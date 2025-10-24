import os
from typing import Tuple, Any

from huggingface_hub import HfFileSystem, hf_hub_url

from core.config import config_utils
from core.config.config_utils import get_settings
from core.model.HfFileInfo import HfFileInfo
from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelInfo import HFModelInfo
from core.model.ModelMetadata import SimpleModelMetadata
from core.model.ModelType import ModelType
from core.model.storage_helpers.HuggingfaceFileSystem import HuggingfaceFileSystem
from core.model.storage_helpers.PullSupplier import PullSupplier, Supplier
from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper

DEFAULT_CONTENT_TYPE = "text/plain"

class RKPullSupplier(PullSupplier):

    def file_info(self, model_name, file, repo, model_type, supplier: Supplier) -> Tuple[
        Any | None, str | None, ModelType | None, Any]:
        if supplier != Supplier.HUGGINGFACE:
            return None, None, None, "Error: Invalid supplier.\n"
        """
        model_file_info={
            "model_name":"Qwen3-1.7B-rk3588-1.2.1-unsloth-16k",
            "model_type":"RKLLM",
            "endpoint_model_file":"Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm",
            "endpoint_model_file_size":2391955766,
            "license":null,
            "huggingface_path":"dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k",
            "ollama_path":null,
            "system_prompt":""
            }
        """
        huggingface_file_info_path = \
            RkllamaStorageHelper.huggingface_file_info_path_from_raw(
                model_name=model_name,
                endpoint_model_file=file,
            )
        hf_file_info: HfFileInfo | None = None

        self.logger.debug(f"huggingface_file_info_path={huggingface_file_info_path}")
        if os.path.exists(huggingface_file_info_path):
            try:
                hf_file_info: HfFileInfo = HfFileInfo.load(file_path=huggingface_file_info_path)
            except Exception as e:
                self.logger.error(f"Error reading model HF file info: {str(e)}")

        # Use Hugging Face HfFileSystem to get the file metadata
        try:
            if hf_file_info is None:
                fs = HfFileSystem()
                file_info = fs.info(repo + "/" + file)

                hf_file_info = HfFileInfo(**file_info)
            self.logger.debug(f"file_info={hf_file_info}")

            """
            file_info: {
                name = 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k/Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm'
                size = 2391955766
                type = 'file'
                blob_id = '2e96f326e6c2c147b1709405b4bd7ee47a6ac94d'
                lfs = BlobLfsInfo(size=2391955766,
                              sha256='27ae60300386eb5e825976dd3346bd6329e0effb7948961e10c90aefc003874c',
                              pointer_size=135)
                last_commit = None
                security = None
            """

            return hf_file_info, repo, model_type, None
        except Exception as e:
            return None, None, None, f"Error: {str(e)}\n"

    def check_file_info(self, model_name, file, repo, model_type, file_info) -> Tuple[Any | None, Any]:
        if (not file_info.name == f"{repo}/{file}") or (not file_info.size == file_info.lfs.size):
            return None, "Error: incorrect HF file info.\n"

        return file_info, None

    def size_and_digest(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any, Any]:
        total_size = file_info.size  # File size in bytes
        if total_size == 0:
            return None, None, "Error: Unable to retrieve file size.\n"

        digest = file_info.lfs.sha256
        if not digest:
            return None, None, "Error: Unable to retrieve file digest.\n"

        return total_size, digest, None

    def model_file_info(self, model_name, file, repo, model_type, file_info: HfFileInfo, supplier: Supplier) -> Tuple[
        Any | None, Any | None, str | None, ModelType | None, Any]:
        if supplier != Supplier.HUGGINGFACE:
            return None, None, None, None, "Error: Invalid supplier.\n"

        # when use Hugging Face HfFileSystem to get the file metadata, have:
        # model_name=Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # file=Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm,
        # repo=dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # supplier=Supplier.HUGGINGFACE

        fs = HuggingfaceFileSystem()

        huggingface_model_info_path = \
            file_info.huggingface_model_info_path
        huggingface_model_info: HFModelInfo | None = None

        self.logger.debug(f"model_file_info(): huggingface_model_info_path={huggingface_model_info_path}")
        if os.path.exists(huggingface_model_info_path):
            try:
                huggingface_model_info = HFModelInfo.load(file_path=huggingface_model_info_path)
            except Exception as e:
                self.logger.error(f"Error reading model huggingface manifest: {str(e)}")

        try:
            if huggingface_model_info is None:
                hf_model_info = fs.load_model_info(huggingface_path=repo)
                self.logger.debug(f"model_file_info(): hf_model_info={hf_model_info}")

                huggingface_model_info = HFModelInfo(**hf_model_info)
            self.logger.debug(f"model_file_info(): huggingface_model_info={huggingface_model_info}")

            return huggingface_model_info, file_info, repo, model_type, None
        except Exception as e:
            self.logger.exception(f"Error reading model HFModelInfo: {str(e)}", exc_info=e)
            return None, None, None, None, f"Error: {str(e)}\n"

    def check_model_file_info(self, model_name, file, repo, model_type, file_info: HfFileInfo, model_file_info: HFModelInfo) -> Tuple[
        Any | None, Any]:
        endpoint_model_file = file_info.name.split("/")[-1]
        model_repo = file_info.name.replace(endpoint_model_file, "").strip("/")
        if model_repo != model_file_info.id:
            return None, "Error: invalid model file info id.\n"
        if model_repo != model_file_info.modelId:
            return None, "Error: invalid model file info modelId.\n"

        rfilename_siblings = [sibling.rfilename for sibling in model_file_info.siblings]
        if not endpoint_model_file in rfilename_siblings:
            return None, "Error: missing endpoint model file in siblings.\n"

        # TODO: check that model_file_info is valid

        return model_file_info, None


    def create_generic_model_file_info(self, file: str, model_name: str, model_type: ModelType | None, repo: str,
                                       supplier: Supplier,
                                       total_size: int,
                                       file_info: HfFileInfo,
                                       model_file_info: HFModelInfo) -> ModelFileInfo:
        """
        fulfill the ModelFileInfo fields that are not derived from Hugging Face HfFileSystem
        ModelFileInfo:
            model_name: str
            model_type: Optional[ModelType] = None
            endpoint_model_file: str
            endpoint_model_file_size: int
            license: Optional[ModelLicense] = None
            huggingface_path: Optional[str] = Field(default=None, description="Hugging Face repository path")
            ollama_path: Optional[str] = Field(default=None, description="Ollama repository path")
            system_prompt: str = ""
            _simple_model_metadata: SimpleModelMetadata = None
        """
        generic_model_file_info: ModelFileInfo = ModelFileInfo(
            model_name=model_name,
            model_type=model_type,
            endpoint_model_file=file,
            endpoint_model_file_size=total_size,
            huggingface_path=repo,
            ollama_path=None,
        )
        generic_model_file_info.huggingface_file_info = file_info
        generic_model_file_info.huggingface_model_info = model_file_info
        self.logger.debug(f"generic_model_file_info={generic_model_file_info.model_dump_json()}")
        self.logger.debug(f"hf_data={generic_model_file_info.huggingface_model_info}")
        return generic_model_file_info

    def create_generic_model_file(self, generic_model_file_info: ModelFileInfo,
                                  file_info: HfFileInfo, model_file_info: HFModelInfo,
                                  model_type: ModelType | None, repo: str) -> Tuple[ModelFile, ModelFileInfo]:
        generic_model_file_info_dump = generic_model_file_info.model_dump()
        self.logger.debug(f"generic_model_file_info_dump={generic_model_file_info_dump}")
        generic_model_file: ModelFile = ModelFile.create(
            model_file_info=generic_model_file_info,
            default_model_config=get_settings().model)
        self.logger.debug(f"generic_model_file={generic_model_file.model_dump_json()}")
        self.logger.debug(
            f"generic_model_file.simple_model_metadata={generic_model_file.simple_model_metadata.model_dump_json()}")
        generic_model_file.huggingface_model_info = model_file_info
        generic_model_file.huggingface_file_info = file_info
        generic_model_file_info = ModelFileInfo(**generic_model_file_info_dump)
        if isinstance(generic_model_file.model_metadata, SimpleModelMetadata):
            generic_model_file_info.simple_model_metadata = generic_model_file.model_metadata
        generic_model_file_info.huggingface_model_info = model_file_info
        generic_model_file_info.huggingface_file_info = file_info
        return generic_model_file, generic_model_file_info

    def lock_model(self, model_file) -> Tuple[int | None, Any]:
        if model_file.is_locked():
            return None, "Error: Model is currently locked.\n"

        lock_id = model_file.lock_model()
        return lock_id, None

    def model_download_url(self, model_name, file, repo, model_type, file_info) -> Tuple[Any, Any]:
        url = hf_hub_url(repo_id=repo, filename=file)
        return url, None

    @property
    def content_type(self) -> str:
        return DEFAULT_CONTENT_TYPE

    def format_progress(self, digest: str, progress: int, total: int, completed: int) -> Any:
        return f"{progress}%\n"

    def format_success(self, digest: str) -> Any:
        return f"{digest}\n"

    def format_error(self, error: str) -> Any:
        return error

