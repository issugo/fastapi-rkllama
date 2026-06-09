import os
import re
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Any, Tuple

from pydantic import Field

from core.config.warnings import deprecated
from core.model import logger
from core.model.ModelName import ModelName, ModelNameException
from core.model.ModelType import ModelType
from core.model.converter.quantization_constants import quant_patterns, quant_mapping
from core.model.models_constants import (
    MODELFILE_NAME,
    B_PARAM_SIZE_PATTERN,
    UNKNOWN_VAL_STR,
    M_PARAM_SIZE_PATTERN,
    validate_model_id,
)

from core.model.HfFileInfo import HfFileInfo
from core.model.OllamaManifest import OllamaManifest
from core.model.suppliers_model_info import HFModelInfo, OllamaModelInfo

MODEL_METADATA_NAME = ".metadata"


class ModelException(Exception):
    pass


class ModelDirError(str, Enum):
    NOT_EXIST = "Model directory not found."
    INVALID = "Model directory not a directory."


class ModelDirException(ModelException):
    model_dir_error: ModelDirError

    def __init__(self, model_dir_error: ModelDirError):
        super().__init__(model_dir_error.value)
        self.model_dir_error = model_dir_error


class ModelNotFoundException(ModelException):
    model_name: str

    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found.")
        self.model_name = model_name


class ModelPath(ModelName):
    endpoint_model_file: str
    endpoint_model_file_size: int

    huggingface_path: Optional[str] = Field(
        default=None, description="Hugging Face repository path"
    )
    ollama_path: Optional[str] = Field(
        default=None, description="Ollama repository path"
    )

    _huggingface_file_info: Union[HfFileInfo | None] = None
    _huggingface_model_info: Union[HFModelInfo | None] = None
    _ollama_file_info: Union[OllamaManifest | None] = None
    _ollama_model_info: Union[OllamaModelInfo | None] = None

    @classmethod
    def from_model_id(cls, model_id: str) -> Any:
        model_id = validate_model_id(model_id=model_id)
        try:
            model_name: ModelName = ModelName.from_model_id(model_id)
            endpoint_model_file = model_id[len(model_name.model_name) + 1 :]
            endpoint_model_file_path: Path = (
                model_name.endpoint_model_file_path_with_endpoint(endpoint_model_file)
            )
            if (
                endpoint_model_file_path.is_file()
                or endpoint_model_file_path.is_symlink()
            ):
                endpoint_model_file_size = endpoint_model_file_path.stat().st_size
                model_path: ModelPath = ModelPath(
                    model_name=model_name.model_name,
                    endpoint_model_file=endpoint_model_file,
                    endpoint_model_file_size=endpoint_model_file_size,
                )
                if model_name.model_format:
                    model_path.model_format = model_name.model_format
                return model_path
            raise ModelNotFoundException(model_name.model_name)
        except ModelNameException as e:
            raise ModelNotFoundException(model_id) from e

    @staticmethod
    def compute_model_id(
        model_name: str, endpoint_model_file: str, is_ollama: bool = False
    ) -> str:
        if is_ollama:
            return validate_model_id(f"{model_name}:{endpoint_model_file}")
        return validate_model_id(f"{model_name}/{endpoint_model_file}")

    @property
    def model_id(self):
        return validate_model_id(
            self.compute_model_id(
                model_name=self.model_name,
                endpoint_model_file=self.endpoint_model_file,
                is_ollama=self.ollama_path is not None,
            )
        )

    def validate_FROM_with_endpoint_file(self, FROM: str) -> bool | Any:
        return (
            self.endpoint_model_file is not None
            and not FROM.endswith(self.endpoint_model_file)
            and FROM.replace(":", "/") != self.model_id.replace(":", "/")
        )

    @property
    def endpoint_model_file_path(self) -> Path:
        return self.endpoint_model_file_path_with_endpoint(self.endpoint_model_file)

    @property
    def model_type(self) -> ModelType | None:
        for mtype in ModelType:
            try:
                if self.endpoint_model_file.endswith(mtype.get_extension()):
                    self.model_format = mtype
                    return mtype
            except AttributeError as e:
                print(
                    "DEBUG model_type AttributeError: self =",
                    self,
                    type(self),
                    "dict =",
                    self.__dict__,
                )
                raise e
        if self.model_format is not None:
            return self.model_format
        # use metadata file if available
        if self.model_metadata_path.exists():
            from core.model.ModelMetadata import SimpleModelMetadata

            model_metadata: SimpleModelMetadata = SimpleModelMetadata.load(
                model_path=self
            )
            mtype: ModelType = model_metadata.model_type
            self.model_format = mtype
            return mtype
        return None

    @deprecated("doesn't take care of new sha storage")
    @property
    def model_exists(self) -> bool:
        if self.model_dir_path.exists():
            if (
                self.endpoint_model_file_path.is_file()
                or self.endpoint_model_file_path.is_symlink()
            ):
                return self.model_type is not None
        return False

    def model_metadata_path_using_model_dir_path(self, model_dir_path: Path) -> Path:
        return model_dir_path / f"{self.endpoint_model_file}{MODEL_METADATA_NAME}"

    @property
    def model_metadata_path(self) -> Path:
        return self.model_metadata_path_using_model_dir_path(self.model_dir_path)

    @staticmethod
    def __modelfile_path(model_name: str) -> Path:
        return Path(ModelPath.model_dir_using_model_name(model_name)) / MODELFILE_NAME

    @property
    def modelfile_path(self) -> Path:
        return self.model_dir_path / MODELFILE_NAME

    @staticmethod
    def __modelfile_exists(model_name: str) -> bool:
        if Path(ModelPath.model_dir_using_model_name(model_name)).exists():
            return ModelPath.__modelfile_path(model_name).exists()
        return False

    @property
    def modelfile_exists(self) -> bool:
        if self.model_dir_path.exists():
            return self.modelfile_path.is_file()
        return False

    @property
    def modelfile_match(self) -> bool:
        if self.modelfile_exists:
            with open(self.modelfile_path, "r") as f:
                for line in f.readlines():
                    if line.startswith("FROM "):
                        mfile_endpoint_model_file = line.split(" ", maxsplit=1)[
                            1
                        ].strip()
                        if mfile_endpoint_model_file.endswith(self.endpoint_model_file):
                            return True
        return False

    def get_model_format(self) -> str:
        model_type: ModelType = self.model_type
        if model_type is None:
            return UNKNOWN_VAL_STR
        return model_type.value.lower()

    def get_model_family(self) -> str:
        from core.model.ModelMetadata import SimpleModelMetadata

        model_metadata, _, _, _ = SimpleModelMetadata.parse_splitted_for_model_family(
            splitted=self.model_name.split("-"),
            start_pos=0,
            model_metadata={},
            model_details={},
            model_tags=[],
        )
        if model_metadata:
            return model_metadata.get("model_family", UNKNOWN_VAL_STR)
        return UNKNOWN_VAL_STR

    @staticmethod
    def get_parameter_size(model_name: str) -> str | None:
        # Extract parameter size (e.g., 3B, 7B, 13B)
        b_param_size_match = re.search(B_PARAM_SIZE_PATTERN, model_name)
        if b_param_size_match:
            size = b_param_size_match.group(1)
            # Convert to standard format (3B, 7B, 13B, etc)
            if "." in size:
                # For sizes like 1.1B, 2.7B
                return f"{size}B"
            else:
                # For sizes like 3B, 7B
                return f"{size}B"
        m_param_size_match = re.search(M_PARAM_SIZE_PATTERN, model_name)
        if m_param_size_match:
            size = m_param_size_match.group(1)
            # Convert to standard format (3M, 7M, 13M, etc)
            if "." in size:
                # For sizes like 1.1M, 2.7M
                return f"{size}M"
            else:
                # For sizes like 3M, 7M
                return f"{size}M"
        return None

    @staticmethod
    def get_ollama_quant_level(model_name: str) -> str | None:
        for quant_type, pattern in quant_patterns:
            if re.search(pattern, model_name, re.IGNORECASE):
                # Use Ollama-style quantization name if available
                return quant_mapping.get(quant_type)
        return None

    @property
    def huggingface_file_info_exists(self) -> bool:
        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper

        if self._huggingface_file_info:
            return True
        huggingface_file_info_path = RkllamaStorageHelper.huggingface_file_info_path(
            self
        )
        if huggingface_file_info_path is None:
            return False
        logger.debug(f"Checking if HF file info exists: {huggingface_file_info_path}")
        if os.path.exists(huggingface_file_info_path):
            if self.huggingface_path is None:
                import json

                self.huggingface_path = json.loads(
                    huggingface_file_info_path.read_text()
                ).get("name")
            return self.huggingface_file_info is not None
        return False

    @property
    def huggingface_file_info(self) -> HfFileInfo | None:
        if not self.huggingface_path or "/" not in self.huggingface_path:
            raise ValueError(f"Invalid huggingface path {self.huggingface_path}")

        if self._huggingface_file_info:
            return self._huggingface_file_info

        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper

        huggingface_file_info_path = RkllamaStorageHelper.huggingface_file_info_path(
            self
        )
        if huggingface_file_info_path is None:
            return None

        try:
            if os.path.exists(huggingface_file_info_path):
                self._huggingface_file_info = HfFileInfo.load(
                    huggingface_file_info_path
                )
                return self._huggingface_file_info
        except Exception as e:
            logger.exception(f"Error loading Huggingface file info: {str(e)}")
            return None

        # else...
        try:
            from core.model.storage_helpers.SupplierFileInfo import Supplier
            from core.model.storage_helpers.RKPullSupplier import RKPullSupplier

            huggingface_pull_supplier: RKPullSupplier = RKPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.HUGGINGFACE
            file_info, repo, model_type, error = huggingface_pull_supplier.file_info(
                model_name=self.model_name,
                file=self.endpoint_file_file,
                repo=None,
                model_type=self.model_type,
                supplier=Supplier.HUGGINGFACE,
            )

            if error:
                logger.debug(f"Failed to get HUGGINGFACE data: {error}")
                return None

            self._huggingface_file_info = file_info
            self._huggingface_file_info.save(
                RkllamaStorageHelper.huggingface_file_info_path(self)
            )
            return self._huggingface_file_info
        except Exception as e:
            logger.exception(f"Error fetching HUGGINGFACE model info: {str(e)}")
            return None

    @huggingface_file_info.setter
    def huggingface_file_info(self, value: HfFileInfo):
        self._huggingface_file_info = value

    @property
    def huggingface_model_info_exists(self) -> bool:
        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper

        if self._huggingface_model_info:
            return True
        huggingface_model_info_path = RkllamaStorageHelper.huggingface_model_info_path(
            self
        )
        if huggingface_model_info_path is None:
            return False
        if os.path.exists(huggingface_model_info_path):
            return self.huggingface_model_info is not None
        return False

    @property
    def huggingface_model_info(self) -> HFModelInfo | None:
        if not self.huggingface_path or "/" not in self.huggingface_path:
            raise ValueError("Invalid HuggingFace path")

        if self._huggingface_model_info:
            return self._huggingface_model_info

        try:
            from core.model.storage_helpers.RkllamaStorageHelper import (
                RkllamaStorageHelper,
            )

            if os.path.exists(RkllamaStorageHelper.huggingface_model_info_path(self)):
                self._huggingface_model_info = HFModelInfo.load(
                    file_path=RkllamaStorageHelper.huggingface_model_info_path(self)
                )
                return self._huggingface_model_info
        except Exception as e:
            logger.exception(f"Error loading HF model info: {str(e)}")
            return None

        # else...
        try:
            from core.model.storage_helpers.SupplierFileInfo import Supplier
            from core.model.storage_helpers.RKPullSupplier import RKPullSupplier

            rk_pull_supplier: RKPullSupplier = RKPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.OLLAMA

            model_file_info, file_info, repo, model_type, error = (
                rk_pull_supplier.model_file_info(
                    model_name=self.model_name,
                    file=self.endpoint_model_file,
                    repo=None,
                    model_type=self.model_type,
                    file_info=self.huggingface_file_info,
                    supplier=Supplier.HUGGINGFACE,
                )
            )

            if error:
                logger.debug(f"Failed to get HUGGINGFACE data: {error}")
                return None

            self._huggingface_model_info = model_file_info
            self._huggingface_model_info.save(
                RkllamaStorageHelper.huggingface_model_info_path(self)
            )
            return self._huggingface_model_info
        except Exception as e:
            logger.exception(f"Error fetching HUGGINGFACE model info: {str(e)}")
            return None

    @huggingface_model_info.setter
    def huggingface_model_info(self, value: HFModelInfo):
        self._huggingface_model_info = value

    @property
    def ollama_file_info_exists(self) -> bool:
        from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper

        if self._ollama_file_info:
            return True
        ollama_file_info_path = OllamaStorageHelper.ollama_file_info_path(self)
        if ollama_file_info_path is None:
            return False
        logger.debug(f"Checking if Ollama file info exists: {ollama_file_info_path}")
        if os.path.exists(ollama_file_info_path):
            if self.ollama_path is None:
                import json

                if (
                    json.loads(ollama_file_info_path.read_text())
                    .get("config", {})
                    .get("digest", "")
                    .startswith("sha256:")
                ):
                    self.ollama_path = self.compute_model_id(
                        model_name=self.model_name,
                        endpoint_model_file=self.endpoint_model_file,
                        is_ollama=True,
                    )
            return self.ollama_file_info is not None
        return False

    @property
    def ollama_file_info(self) -> OllamaManifest | None:
        if not self.ollama_path or ":" not in self.ollama_path:
            raise ValueError(f"Invalid ollama path {self.ollama_path}")

        if self._ollama_file_info:
            return self._ollama_file_info

        from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper

        ollama_file_info_path = OllamaStorageHelper.ollama_file_info_path(self)
        if ollama_file_info_path is None:
            return None

        try:
            if os.path.exists(ollama_file_info_path):
                self._ollama_file_info = OllamaManifest.load(ollama_file_info_path)
                return self._ollama_file_info
        except Exception as e:
            logger.exception(f"Error loading Ollama file info: {str(e)}")
            return None

        # else...
        try:
            from core.model.storage_helpers.SupplierFileInfo import Supplier
            from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier

            ollama_pull_supplier: OllamaPullSupplier = OllamaPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.OLLAMA
            file_info, repo, model_type, error = ollama_pull_supplier.file_info(
                model_name=self.model_name,
                file=self.endpoint_file_file,
                repo=None,
                model_type=self.model_type,
                supplier=Supplier.OLLAMA,
            )

            if error:
                logger.debug(f"Failed to get OLLAMA data: {error}")
                return None

            self._ollama_file_info = file_info
            self._ollama_file_info.save(OllamaStorageHelper.ollama_file_info_path(self))
            return self._ollama_file_info
        except Exception as e:
            logger.exception(f"Error fetching OLLAMA model info: {str(e)}")
            return None

    @ollama_file_info.setter
    def ollama_file_info(self, value: OllamaManifest):
        self._ollama_file_info = value

    @property
    def ollama_model_info_exists(self) -> bool:
        from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper

        if self._ollama_model_info:
            return True
        ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(self)
        if ollama_model_info_path is None:
            return False
        if isinstance(ollama_model_info_path, str):
            ollama_model_info_path = Path(ollama_model_info_path)
        if ollama_model_info_path.exists():
            return self.ollama_model_info is not None
        return False

    @property
    def ollama_model_info(self) -> OllamaModelInfo | None:
        if not self.ollama_path or ":" not in self.ollama_path:
            raise ValueError("Invalid ollama path")

        if self._ollama_model_info:
            return self._ollama_model_info

        from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper

        ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(self)
        if ollama_model_info_path is None:
            return None

        try:
            if isinstance(ollama_model_info_path, str):
                ollama_model_info_path = Path(ollama_model_info_path)
            if ollama_model_info_path.exists():
                self._ollama_model_info = OllamaModelInfo.load(ollama_model_info_path)
                return self._ollama_model_info
        except Exception as e:
            logger.exception(f"Error loading Ollama model info: {str(e)}")
            return None

        # else...
        try:
            from core.model.storage_helpers.SupplierFileInfo import Supplier
            from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier

            ollama_pull_supplier: OllamaPullSupplier = OllamaPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.OLLAMA

            model_file_info, file_info, repo, model_type, error = (
                ollama_pull_supplier.model_file_info(
                    model_name=self.model_name,
                    file=self.endpoint_model_file,
                    repo=None,
                    model_type=self.model_type,
                    file_info=self.ollama_file_info,
                    supplier=Supplier.OLLAMA,
                )
            )

            if error:
                logger.debug(f"Failed to get OLLAMA data: {error}")
                return None

            self._ollama_model_info = model_file_info
            self._ollama_model_info.save(
                OllamaStorageHelper.ollama_model_info_path(self)
            )
            return self._ollama_model_info
        except Exception as e:
            logger.exception(f"Error fetching OLLAMA model info: {str(e)}")
            return None

    @ollama_model_info.setter
    def ollama_model_info(self, value: OllamaModelInfo):
        self._ollama_model_info = value

    @property
    def repo_url(self):
        if self.huggingface_path:
            from core.model.storage_helpers.HuggingfaceFileSystem import (
                HuggingfaceFileSystem,
            )

            return HuggingfaceFileSystem.model_path(self.huggingface_path)
        elif self.ollama_path:
            from core.model.storage_helpers.OllamaFileSystem import OllamaFileSystem

            return OllamaFileSystem.model_path(self.ollama_path, api=False)
        else:
            return None

    def lock_model(self) -> int:
        """Lock the model to prevent concurrent access."""
        # Create output directory with model name
        os.makedirs(self.model_dir, exist_ok=True)
        lock_file = os.path.join(self.model_dir, "lock")
        # TODO: check if lock_file exists then return -1
        try:
            with open(lock_file, "w") as f:
                f.write(str(os.getpid()))
            return os.getpid()
        except Exception as e:
            logger.exception(f"Error locking model: {str(e)}")
            return -1

    def unlock_model(self, lock_id: int) -> None:
        """Unlock the model to allow concurrent access."""
        lock_file = os.path.join(self.model_dir, "lock")
        # TODO: test that lock_file contains lock_id
        try:
            os.remove(lock_file)
            logger.info(f"Unlocked model {self.model_name} with lock ID {lock_id}")
        except Exception as e:
            logger.exception(f"Error unlocking model: {str(e)}")

    def is_locked(self) -> bool:
        """Check if the model is locked."""
        lock_file = os.path.join(self.model_dir, "lock")
        return os.path.exists(lock_file)


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def validate_model_path(path: str) -> bool:
    """Validate if a path points to a valid model file."""
    if not os.path.exists(path):
        return False
    if not os.path.isfile(path):
        return False
    return True


def find_rkllm_model_name(model_dir):
    """
    Find the RKLLM model name based on the model dir.

    Args:
        model_dir: Directory of the model (can be simplified or full path)

    Returns:
        The name to the RKLLM model or None if not found
    """
    for file in os.listdir(model_dir):
        if file.endswith(".rkllm") and os.path.isfile(os.path.join(model_dir, file)):
            return file
    return None


def str_parameters_size(content: int) -> Tuple[float, str, int]:
    if content > 1_000_000_000:
        return float(content) / 1_000_000_000, "B", content
    if content > 1_000_000:
        return float(content) / 1_000_000, "M", content
    raise ValueError()


def int_parameters_size(content: str) -> Tuple[float, str, int]:
    b_param_pattern = re.search(
        B_PARAM_SIZE_PATTERN,
        content,
    )
    if b_param_pattern:
        int_size_value = None
        size_value = float(b_param_pattern.group(1))
        size_unit = b_param_pattern.group(2).lower()
        # Convert to billions if needed
        if size_unit == "b" or size_unit == "B":
            int_size_value = int(size_value * 1_000_000_000)
        return size_value, size_unit.upper(), int_size_value
    m_param_pattern = re.search(
        M_PARAM_SIZE_PATTERN,
        content,
    )
    if m_param_pattern:
        int_size_value = None
        size_value = float(m_param_pattern.group(1))
        size_unit = m_param_pattern.group(2).lower()
        # Convert to billions if needed
        if size_unit == "m" or size_unit == "M":
            int_size_value = int(size_value * 1_000_000)
        return size_value, size_unit.upper(), int_size_value
    else:
        raise ValueError()
