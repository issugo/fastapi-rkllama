import datetime
import os
from typing import Tuple, Any

import requests
from huggingface_hub import HfFileSystem, hf_hub_url

from core.config.config_utils import get_settings
from core.model.HfFileInfo import HfFileInfo
from core.model.Model import Model
from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelInfo import ModelInfo, DummyStatResult
from core.model.ModelMetadata import SimpleModelMetadata, create_metadata
from core.model.ModelPath import ModelPath
from core.model.ModelType import ModelType
from core.model.models_constants import validate_model_id
from core.model.storage_helpers.HuggingfaceFileSystem import HuggingfaceFileSystem
from core.model.storage_helpers.PullSupplier import PullSupplier
from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper
from core.model.storage_helpers.SupplierFileInfo import Supplier
from core.model.suppliers_model_info import HFModelInfo, HFModelLicense

DEFAULT_CONTENT_TYPE = "text/plain"


class RKPullSupplier(PullSupplier):
    def file_info(
        self, model_name, file, repo, model_type, supplier: Supplier
    ) -> Tuple[Any | None, str | None, ModelType | None, Any]:
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
        huggingface_file_info_path = (
            RkllamaStorageHelper.huggingface_file_info_path_from_raw(
                model_name=model_name,
                endpoint_model_file=file,
            )
        )
        hf_file_info: HfFileInfo | None = None

        self.logger.debug(f"huggingface_file_info_path={huggingface_file_info_path}")
        if os.path.exists(huggingface_file_info_path):
            try:
                hf_file_info: HfFileInfo = HfFileInfo.load(
                    file_path=huggingface_file_info_path
                )
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

    def check_file_info(
        self, model_name, file, repo, model_type, file_info
    ) -> Tuple[Any | None, Any]:
        if (not file_info.name == f"{repo}/{file}") or (
            not file_info.size == file_info.lfs.size
        ):
            return None, "Error: incorrect HF file info.\n"

        return file_info, None

    def model_file_info(
        self,
        model_name,
        file,
        repo,
        model_type,
        file_info: HfFileInfo,
        supplier: Supplier,
    ) -> Tuple[Any | None, Any | None, str | None, ModelType | None, Any]:
        if supplier != Supplier.HUGGINGFACE:
            return None, None, None, None, "Error: Invalid supplier.\n"

        # when use Hugging Face HfFileSystem to get the file metadata, have:
        # model_name=Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # file=Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm,
        # repo=dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
        # supplier=Supplier.HUGGINGFACE

        fs = HuggingfaceFileSystem()

        huggingface_model_info_path = file_info.huggingface_model_info_path
        huggingface_model_info: HFModelInfo | None = None

        self.logger.debug(
            f"model_file_info(): huggingface_model_info_path={huggingface_model_info_path}"
        )
        if os.path.exists(huggingface_model_info_path):
            try:
                huggingface_model_info = HFModelInfo.load(
                    file_path=huggingface_model_info_path
                )
            except Exception as e:
                self.logger.error(f"Error reading model huggingface manifest: {str(e)}")

        try:
            if huggingface_model_info is None:
                hf_model_info = fs.load_model_info(huggingface_path=repo)
                self.logger.debug(f"model_file_info(): hf_model_info={hf_model_info}")

                huggingface_model_info = HFModelInfo(**hf_model_info)
            self.logger.debug(
                f"model_file_info(): huggingface_model_info={huggingface_model_info}"
            )

            # manage license
            rfilename_siblings = [
                sibling.rfilename for sibling in huggingface_model_info.siblings
            ]
            if "LICENSE" in rfilename_siblings:
                license_url = fs.sibling_url(
                    huggingface_path=repo, rfilename_sibling="LICENSE"
                )
                with requests.get(license_url) as r:
                    huggingface_model_info.license = HFModelLicense.from_content(
                        content=r.content, license_url=license_url
                    )
            elif (
                hf_model_info
                and "license" in hf_model_info
                and "license_name" in hf_model_info
                and "license_url" in hf_model_info
            ):
                huggingface_model_info.license = HFModelLicense(
                    supplier=Supplier.HUGGINGFACE,
                    license_name=hf_model_info["license_name"],
                    license_url=hf_model_info["license_url"],
                    license_text=hf_model_info["license"],
                )

            # manage template: already in huggingface_model_info.config.chat_template_jinja

            return huggingface_model_info, file_info, repo, model_type, None
        except Exception as e:
            self.logger.exception(
                f"Error reading model HFModelInfo: {str(e)}", exc_info=e
            )
            return None, None, None, None, f"Error: {str(e)}\n"

    def check_model_file_info(
        self,
        model_name,
        file,
        repo,
        model_type,
        file_info: HfFileInfo,
        model_file_info: HFModelInfo,
    ) -> Tuple[Any | None, Any]:
        endpoint_model_file = file_info.name.split("/")[-1]
        model_repo = file_info.name.replace(endpoint_model_file, "").strip("/")
        if model_repo != model_file_info.id:
            return None, "Error: invalid model file info id.\n"
        if model_repo != model_file_info.modelId:
            return None, "Error: invalid model file info modelId.\n"

        rfilename_siblings = [sibling.rfilename for sibling in model_file_info.siblings]
        if endpoint_model_file not in rfilename_siblings:
            return None, "Error: missing endpoint model file in siblings.\n"

        # TODO: check that model_file_info is valid

        return model_file_info, None

    def create_generic_model_info(
        self,
        file: str,
        model_name: str,
        model_type: ModelType | None,
        repo: str,
        supplier: Supplier,
        total_size: int,
        digest: str,
        file_info: HfFileInfo,
        model_file_info: HFModelInfo,
    ) -> ModelInfo:
        """
        fulfill the ModelInfo fields that are not derived from Hugging Face HfFileSystem
        """
        model_stat = self._create_dummy_stat_result(file_info, model_file_info)
        generic_model_info: ModelInfo = ModelInfo.from_hf_model_info(
            hf_model_info=model_file_info,
            model_path=ModelPath(
                model_name=model_name,
                model_type=model_type,
                endpoint_model_file=file,
                endpoint_model_file_size=total_size,
            ),
            size=total_size,
            digest=digest,
            model_stat=model_stat,
        )
        self.logger.debug(f"generic_model_info={generic_model_info.model_dump_json()}")
        self.logger.debug(f"hf_data={generic_model_info.hf_model_info}")
        return generic_model_info

    def _create_dummy_stat_result(
        self, file_info: HfFileInfo, model_file_info: HFModelInfo
    ) -> DummyStatResult:
        dt_now = datetime.datetime.now().timestamp()
        dt_modified = (
            datetime.datetime.strptime(
                model_file_info.lastModified, "%Y-%m-%dT%H:%M:%S.%fZ"
            ).timestamp()
            if model_file_info.lastModified
            else (
                datetime.datetime.strptime(
                    HfFileInfo.last_commit_to_last_modified(file_info.last_commit),
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                ).timestamp()
                if file_info.last_commit
                else dt_now
            )
        )
        model_stat = DummyStatResult(
            st_size=file_info.size,
            st_atime=dt_modified,
            st_ctime=dt_modified,
            st_mtime=dt_modified,
        )
        return model_stat

    def create_generic_model(
        self,
        generic_model_info: ModelInfo,
        file_info: HfFileInfo,
        model_file_info: HFModelInfo,
        model_type: ModelType | None,
        repo: str,
    ) -> Tuple[Model, ModelInfo]:
        """
        fulfill the Model fields to create the Model:
            id: str
            st_atime: float
            st_mtime: float
            st_ctime: float
            size: int
            digest: str
            model_path: ModelPath
            # model_info contains only model file stats, and nothing in relation with model content configuration
            model_info: ModelInfo
            # model_metadata contains model configuration
            model_metadata: Optional[BasicModelMetadata|SimpleModelMetadata|ModelMetadata] = Field(default=None, description="Model metadata")

            _supplier: Optional[Supplier] = None
            _supplier_model_info: Optional[OllamaModelInfo|HFModelInfo] = None

        """
        model_path: ModelPath = generic_model_info.model_path
        model_metadata, model_metadata_format, model_metadata_path = create_metadata(
            model_path=model_path, hf_model_info=model_file_info
        )
        model_stat = self._create_dummy_stat_result(file_info, model_file_info)
        model: Model = Model(
            id=validate_model_id(model_path.model_id),
            st_atime=model_stat.st_atime,
            st_mtime=model_stat.st_mtime,
            st_ctime=model_stat.st_ctime,
            size=file_info.size,
            digest=file_info.digest,
            model_path=model_path,
            model_info=generic_model_info,
            model_metadata=model_metadata,
        )
        model._supplier = Supplier.HUGGINGFACE
        model._supplier_model_info = model_file_info
        return model, generic_model_info

    def create_generic_model_file_info(
        self,
        file: str,
        model_name: str,
        model_type: ModelType | None,
        repo: str,
        supplier: Supplier,
        total_size: int,
        file_info: HfFileInfo,
        model_file_info: HFModelInfo,
    ) -> ModelFileInfo:
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
        self.logger.debug(
            f"generic_model_file_info={generic_model_file_info.model_dump_json()}"
        )
        self.logger.debug(f"hf_data={generic_model_file_info.huggingface_model_info}")
        return generic_model_file_info

    def create_generic_model_file(
        self,
        generic_model_file_info: ModelFileInfo,
        model: Model,
        file_info: HfFileInfo,
        model_file_info: HFModelInfo,
        model_type: ModelType | None,
        repo: str,
    ) -> Tuple[ModelFile, ModelFileInfo]:
        generic_model_file_info_dump = generic_model_file_info.model_dump()
        self.logger.debug(
            f"generic_model_file_info_dump={generic_model_file_info_dump}"
        )
        generic_model_file: ModelFile = ModelFile.create(
            model_file_info=generic_model_file_info,
            default_model_config=get_settings().model,
            model=model,
            model_license=model_file_info.license,
        )
        self.logger.debug(f"generic_model_file={generic_model_file.model_dump_json()}")
        self.logger.debug(
            f"generic_model_file.simple_model_metadata={generic_model_file.simple_model_metadata.model_dump_json()}"
        )
        generic_model_file.huggingface_model_info = model_file_info
        generic_model_file.huggingface_file_info = file_info
        generic_model_file_info = ModelFileInfo(**generic_model_file_info_dump)
        if isinstance(generic_model_file.model_metadata, SimpleModelMetadata):
            generic_model_file_info.simple_model_metadata = (
                generic_model_file.model_metadata
            )
        generic_model_file_info.huggingface_model_info = model_file_info
        generic_model_file_info.huggingface_file_info = file_info
        return generic_model_file, generic_model_file_info

    def lock_model(self, model_file) -> Tuple[int | None, Any]:
        if model_file.is_locked():
            return None, "Error: Model is currently locked.\n"

        lock_id = model_file.lock_model()
        return lock_id, None

    def model_download_url(
        self, model_name, file, repo, model_type, file_info
    ) -> Tuple[Any, Any]:
        url = hf_hub_url(repo_id=repo, filename=file)
        return url, None

    @property
    def content_type(self) -> str:
        return DEFAULT_CONTENT_TYPE

    def format_progress(
        self, digest: str, progress: int, total: int, completed: int
    ) -> Any:
        return f"{progress}%\n"

    def format_success(self, digest: str) -> Any:
        return f"{digest}\n"

    def format_error(self, error: str) -> Any:
        return error
