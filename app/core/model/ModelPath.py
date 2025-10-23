import os
import re
from pathlib import Path
from typing import Union, Optional

from pydantic import BaseModel, Field

from core.config import config_utils
from core.model import logger
from core.model.HfFileInfo import HfFileInfo
from core.model.ModelInfo import ModelDetails, HFModelInfo, OllamaModelInfo
from core.model.ModelName import ModelName
from core.model.ModelType import ModelType
from core.model.OllamaManifest import OllamaManifest
from core.model.converter.quantization_constants import quant_patterns, quant_mapping, ollama_quant_mapping
from core.model.models_constants import MODELFILE_NAME, B_PARAM_SIZE_PATTERN, \
    UNKNOWN_VAL_STR, M_PARAM_SIZE_PATTERN


class ModelLicense(BaseModel):
    license_name: str = None
    license_url: str = None
    license_text: str = None


class ModelPath(ModelName):
    endpoint_model_file: str
    endpoint_model_file_size: int
    license: Optional[ModelLicense] = None

    huggingface_path: Optional[str] = Field(default=None, description="Hugging Face repository path")
    ollama_path: Optional[str] = Field(default=None, description="Ollama repository path")

    _model_dir: Union[str | None] = None
    _huggingface_file_info: Union[HfFileInfo | None] = None
    _huggingface_model_info: Union[HFModelInfo | None] = None
    _ollama_file_info: Union[OllamaManifest | None] = None
    _ollama_model_info: Union[OllamaModelInfo | None] = None

    @staticmethod
    def model_dir_using_model_name(model_name: str) -> str:
        return os.path.join(config_utils.rkllama_config.paths.models, model_name)

    @property
    def model_dir(self):
        if not self._model_dir:
            model_ext = self.model_type.get_extension()
            if model_ext:
                default_relative_dir = self.model_name.replace(model_ext, '')
            else:
                default_relative_dir = self.model_name
            self._model_dir = ModelPath.model_dir_using_model_name(model_name=default_relative_dir)
        return self._model_dir

    @property
    def endpoint_model_file_path(self):
        return os.path.join(self.model_dir, self.endpoint_model_file)

    @property
    def model_type(self) -> ModelType | None:
        for mtype in ModelType:
            if self.endpoint_model_file.endswith(mtype.get_extension()):
                return mtype
        if super().model_type is not None:
            return super().model_type
        return None

    @property
    def model_exists(self) -> bool:
        if os.path.exists(self.model_dir):
            if os.path.isfile(os.path.join(self.model_dir, self.endpoint_model_file)):
                if self.model_type is None:
                    self.__setattr__("model_type", ModelType(self.endpoint_model_file.split(".")[-1].upper()))
                return self.model_type.get_extension() == f".{self.endpoint_model_file.split(".")[-1]}"
        return False

    @property
    def modelfile(self):
        return os.path.join(self.model_dir, MODELFILE_NAME)

    @property
    def modelfile_exists(self) -> bool:
        if os.path.exists(self.model_dir):
            return os.path.isfile(self.modelfile)
        return False

    @property
    def modelfile_match(self) -> bool:
        if self.modelfile_exists:
            with open(self.modelfile, "r") as f:
                for line in f.readlines():
                    if line.startswith("FROM="):
                        mfile_endpoint_model_file = line.split("=")[1].strip()
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
        model_metadata, _, _, _ = \
            SimpleModelMetadata.parse_splitted_for_model_family(
                splitted=self.model_name.split("-"), start_pos=0,
                model_metadata={}, model_details={}, model_tags=[])
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

    def extract_model_details(self) -> ModelDetails:
        """
        Extract model parameter size and quantization type from model name

        Args:
            model_name: Model name or file path

        Returns:
            Dictionary with parameter_size and quantization_level
        """
        # Initialize default values
        # TODO: set model_format and model_family, then remove Optionals from OllamaModelDetails
        details = ModelDetails(
            model_format=UNKNOWN_VAL_STR, model_family=UNKNOWN_VAL_STR,
            parameter_size=UNKNOWN_VAL_STR, quantization_level=UNKNOWN_VAL_STR)

        # Remove path and extension if present
        if isinstance(self.model_name, str):
            basename = os.path.basename(self.model_name).replace(self.model_type.get_extension(), "")
        else:
            basename = str(model_name)

        model_format = self.get_model_format()
        if model_format:
            details.model_format = model_format

        model_family = self.get_model_family()
        if model_family:
            details.model_family = model_family

        parameter_size = ModelPath.get_parameter_size(basename)
        if parameter_size:
            details.parameter_size = parameter_size

        ollama_quant_level = ModelPath.get_ollama_quant_level(basename)
        if ollama_quant_level:
            details.quantization_level = ollama_quant_level

        return details

    @staticmethod
    def gen_endpoint_model_file_name_using_model_details(
            model_name: str,
            model_type: ModelType,
            model_details: ModelDetails
    ) -> str:
        endpoint_model_file = model_name
        parameter_size = ModelPath.get_parameter_size(model_name)
        if parameter_size:
            if parameter_size != model_details.parameter_size:
                endpoint_model_file = f"{endpoint_model_file}_{model_details.parameter_size}"
        if model_details.quantization_level:
            endpoint_model_file = f"{endpoint_model_file}_{ollama_quant_mapping.get(model_details.quantization_level)}"
        return f"{endpoint_model_file}.{model_type.get_extension()}"

    @property
    def huggingface_file_info_exists(self) -> bool:
        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper
        if self._huggingface_file_info:
            return True
        huggingface_file_info_path = RkllamaStorageHelper.huggingface_file_info_path(self)
        if huggingface_file_info_path is None:
            return False
        if os.path.exists(huggingface_file_info_path):
            return self.huggingface_file_info is not None
        return False

    @property
    def huggingface_file_info(self) -> HfFileInfo | None:
        if not self.huggingface_path or "/" not in self.huggingface_path:
            raise ValueError("Invalid huggingface path")

        if self._huggingface_file_info:
            return self._huggingface_file_info

        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper

        huggingface_file_info_path = RkllamaStorageHelper.huggingface_file_info_path(self)
        if huggingface_file_info_path is None:
            return None

        try:
            if os.path.exists(huggingface_file_info_path):
                self._huggingface_file_info = HfFileInfo.load(huggingface_file_info_path)
                return self._huggingface_file_info
        except Exception as e:
            logger.exception(f"Error loading Huggingface file info: {str(e)}")
            return None

        # else...
        try:
            from core.model.storage_helpers.PullSupplier import Supplier
            from core.model.storage_helpers.RKPullSupplier import RKPullSupplier

            huggingface_pull_supplier: RKPullSupplier = RKPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.HUGGINGFACE
            file_info, repo, model_type, error = huggingface_pull_supplier.file_info(
                model_name=self.model_name, file=self.endpoint_file_file,
                repo=None, model_type=self.model_type, supplier=Supplier.HUGGINGFACE)

            if error:
                logger.debug(f"Failed to get HUGGINGFACE data: {error}")
                return None

            self._huggingface_file_info = file_info
            self._huggingface_file_info.save(RkllamaStorageHelper.huggingface_file_info_path(self))
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
        huggingface_model_info_path = RkllamaStorageHelper.huggingface_model_info_path(self)
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
            from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper
            if os.path.exists(RkllamaStorageHelper.huggingface_model_info_path(self)):
                self._huggingface_model_info = HFModelInfo.load(
                    file_path=RkllamaStorageHelper.huggingface_model_info_path(self))
                return self._huggingface_model_info
        except Exception as e:
            logger.exception(f"Error loading HF model info: {str(e)}")
            return None

        # else...
        try:
            from core.model.storage_helpers.PullSupplier import Supplier
            from core.model.storage_helpers.RKPullSupplier import RKPullSupplier

            rk_pull_supplier: RKPullSupplier = RKPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.OLLAMA

            model_file_info, file_info, repo, model_type, error = rk_pull_supplier.model_file_info(
                model_name=self.model_name, file=self.endpoint_model_file,
                repo=None, model_type=self.model_type,
                file_info=self.huggingface_file_info,
                supplier=Supplier.HUGGINGFACE)

            if error:
                logger.debug(f"Failed to get HUGGINGFACE data: {error}")
                return None

            self._huggingface_model_info = model_file_info
            self._huggingface_model_info.save(RkllamaStorageHelper.huggingface_model_info_path(self))
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
        if os.path.exists(ollama_file_info_path):
            return self.ollama_file_info is not None
        return False

    @property
    def ollama_file_info(self) -> OllamaManifest | None:
        if not self.ollama_path or "/" not in self.ollama_path:
            raise ValueError("Invalid ollama path")

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
            from core.model.storage_helpers.PullSupplier import Supplier
            from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier

            ollama_pull_supplier: OllamaPullSupplier = OllamaPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.OLLAMA
            file_info, repo, model_type, error = ollama_pull_supplier.file_info(
                model_name=self.model_name, file=self.endpoint_file_file,
                repo=None, model_type=self.model_type, supplier=Supplier.OLLAMA)

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
        if not self.ollama_path or "/" not in self.ollama_path:
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
            from core.model.storage_helpers.PullSupplier import Supplier
            from core.model.storage_helpers.OllamaPullSupplier import OllamaPullSupplier

            ollama_pull_supplier: OllamaPullSupplier = OllamaPullSupplier()

            # model_name=qwen2.5, file=1.5b, repo=None, supplier=Supplier.OLLAMA

            model_file_info, file_info, repo, model_type, error = ollama_pull_supplier.model_file_info(
                model_name=self.model_name, file=self.endpoint_model_file,
                repo=None, model_type=self.model_type,
                file_info=self.ollama_file_info,
                supplier=Supplier.OLLAMA)

            if error:
                logger.debug(f"Failed to get OLLAMA data: {error}")
                return None

            self._ollama_model_info = model_file_info
            self._ollama_model_info.save(OllamaStorageHelper.ollama_model_info_path(self))
            return self._ollama_model_info
        except Exception as e:
            logger.exception(f"Error fetching OLLAMA model info: {str(e)}")
            return None

    @ollama_model_info.setter
    def ollama_model_info(self, value: OllamaModelInfo):
        self._ollama_model_info = value

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


def GetModels():
    print("Retrieving models...")

    MODEL_PATH = os.path.join(rkllama_config.get_path("models"))

    if not os.path.exists(MODEL_PATH):
        print("Models directory did not exist.\nCreating it now...")
        os.mkdir(MODEL_PATH)

    models_list = []

    for dest, flooders, files in os.walk(MODEL_PATH):
        for file in files:
            if file.endswith(".rkllm"):
                models_list.append(file)

    print("Number of valid models:", len(models_list), "\n")

    return models_list


def int_parameters_size(content: str):
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
        return None, None, None
