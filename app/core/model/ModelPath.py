import os
import re
from typing import Union

import requests
from pydantic import BaseModel

from core.config import config_utils
from core.model.ModelInfo import ModelDetails, HFModelInfo, OllamaModelInfo
from core.model.ModelName import ModelName
from core.model.ModelType import ModelType
from core.model import logger
from core.model.converter.quantization_constants import quant_patterns, quant_mapping, ollama_quant_mapping
from core.model.models_constants import LICENSE_NAME_MAPPING, RK_TAGS_LIST, LANGUAGE_DEFAULT, \
    LANGUAGE_MULTILINGUAL_LIST, LANGUAGE_PATTERNS, MODEL_ARCHITECTURES, MODELFILE_NAME, PARAM_SIZE_PATTERN, \
    UNKNOWN_VAL_STR


class ModelLicense(BaseModel):
    license_name: str = None
    license_url: str = None
    license_text: str = None

class ModelPath(ModelName):
    huggingface_path: str
    endpoint_model_file: str
    endpoint_model_file_size: int
    license: BaseModel = None

    _model_dir: Union[str|None] = None
    _huggingface_model_info: Union[HFModelInfo|None] = None
    _ollama_model_info: Union[OllamaModelInfo|None] = None

    @property
    def model_dir(self):
        if not self._model_dir:
            model_ext = self.model_type.get_extension()
            if model_ext:
                default_relative_dir = self.model_name.replace(model_ext, '')
            else:
                default_relative_dir = self.model_name
            self._model_dir = os.path.join(config_utils.rkllama_config.paths.models, default_relative_dir)
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
    def get_parameter_size(model_name: str)-> str | None:
        # Extract parameter size (e.g., 3B, 7B, 13B)
        param_size_match = re.search(PARAM_SIZE_PATTERN, model_name)
        if param_size_match:
            size = param_size_match.group(1)
            # Convert to standard format (3B, 7B, 13B, etc)
            if "." in size:
                # For sizes like 1.1B, 2.7B
                return f"{size}B"
            else:
                # For sizes like 3B, 7B
                return f"{size}B"
        return None

    @staticmethod
    def get_ollama_quant_level(model_name: str)-> str | None:
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
    def huggingface_model_info_exists(self) -> bool:
        from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper
        if self._huggingface_model_info:
            return True
        elif os.path.exists(RkllamaStorageHelper.huggingface_model_info_path(self)):
            return self.huggingface_model_info is not None
        return False

    @property
    def huggingface_model_info(self) -> HFModelInfo | None:
        """
        Fetch model metadata from Hugging Face API if available.

        Args:
            model_path: HuggingFace repository path (e.g., 'c01zaut/Qwen2.5-3B-Instruct-RK3588-1.1.4')

        Returns:
            Dictionary with enhanced model metadata or None if not available
        """
        if not self.huggingface_path or "/" not in self.huggingface_path:
            raise ValueError("Invalid HuggingFace path")

        if self._huggingface_model_info:
            return self._huggingface_model_info

        try:
            from core.model.storage_helpers.RkllamaStorageHelper import RkllamaStorageHelper
            if os.path.exists(RkllamaStorageHelper.huggingface_model_info_path(self)):
                self._huggingface_model_info = HFModelInfo.load(RkllamaStorageHelper.huggingface_model_info_path(self))
                return self._huggingface_model_info
        except Exception as e:
            logger.exception(f"Error loading HF model info: {str(e)}")
            return None

        # else...
        try:
        # Extract repo_id from HUGGINGFACE_PATH
            url = f"https://huggingface.co/api/models/{self.huggingface_path}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                hf_data = response.json()

                # Process and enhance the metadata
                if "tags" not in hf_data:
                    hf_data["tags"] = []

                # Extract additional info from readme if available
                if "cardData" not in hf_data:
                    hf_data["cardData"] = {}

                # Try to extract parameter size from model name if not in cardData
                if "params" not in hf_data["cardData"]:
                    # Look for patterns like "7b", "3B", "1.5B" in model name or description
                    size_value, size_unit, int_size_value = \
                        int_parameters_size(content=self.huggingface_path + " " + (hf_data.get("description") or ""))
                    hf_data["cardData"]["params"] = int(int_size_value)

                # Extract important information from the description
                description = hf_data.get("description", "")
                if description:
                    # Look for model details in the description
                    quant_pattern = re.search(
                        r"([qQ]\d+_\d+|int4|int8|fp16|4bit|8bit)", description
                    )
                    if quant_pattern:
                        hf_data["quantization"] = quant_pattern.group(1)

                    for arch_name, arch_value in MODEL_ARCHITECTURES.items():
                        if arch_name.lower() in description.lower():
                            hf_data["architecture"] = arch_value
                            if arch_name.lower() not in hf_data["tags"]:
                                hf_data["tags"].append(arch_name.lower())

                # Try to extract language information
                languages = []

                for lang_name, lang_code in LANGUAGE_PATTERNS.items():
                    if (
                            lang_name.lower() in description.lower()
                            or lang_name.lower() in " ".join(hf_data["tags"]).lower()
                    ):
                        if lang_name == "multilingual":
                            # For multilingual models, add common languages
                            languages.extend(LANGUAGE_MULTILINGUAL_LIST)
                        elif lang_code and lang_code not in languages:
                            languages.append(lang_code)

                # If we found languages, add them
                if languages:
                    hf_data["languages"] = list(set(languages))  # Remove duplicates
                elif "en" not in hf_data.get("languages", []):
                    # Default to English if no languages detected
                    hf_data["languages"] = LANGUAGE_DEFAULT

                # Add RK tags if they exist
                rk_patterns = RK_TAGS_LIST
                for pattern in rk_patterns:
                    if (
                            pattern in self.huggingface_path.lower()
                            or pattern in " ".join(hf_data["tags"]).lower()
                            or pattern in description.lower()
                    ):
                        if "rockchip" not in hf_data["tags"]:
                            hf_data["tags"].append("rockchip")
                        if pattern not in hf_data["tags"] and pattern != "rockchip":
                            hf_data["tags"].append(pattern)

                # Add metadata about model capabilities
                if 'sibling_models' in hf_data:
                    for sibling in hf_data.get('sibling_models', []):
                        if sibling.get('rfilename', '').endswith('.rkllm'):
                            hf_data['has_rkllm'] = True
                            break

                # Extract license information
                if "license" in hf_data and hf_data["license"]:

                    license_id = hf_data["license"].lower()
                    hf_data["license_name"] = LICENSE_NAME_MAPPING.get(license_id, hf_data["license"])
                    hf_data["license_url"] = (
                        f"https://huggingface.co/{self.huggingface_path}/blob/main/LICENSE"
                    )

                logger.debug(f"Enhanced model info from HF API: {self.huggingface_path}={hf_data}")

                self._huggingface_model_info = HFModelInfo(**hf_data)
                self._huggingface_model_info.save(RkllamaStorageHelper.huggingface_model_info_path(self))
                return self._huggingface_model_info
            else:
                logger.debug(f"Failed to get HF data: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Error fetching HF model info: {str(e)}")
            return None

    @property
    def ollama_model_info_exists(self) -> bool:
        from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper
        if self._ollama_model_info:
            return True
        ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(self)
        if ollama_model_info_path is None:
            return False
        if os.path.exists(ollama_model_info_path):
            return self.ollama_model_info is not None
        return False

    @property
    def ollama_model_info(self) -> OllamaModelInfo | None:
        """
        Fetch model metadata from Hugging Face API if available.

        Args:
            model_path: ollama repository path (e.g., 'c01zaut/Qwen2.5-3B-Instruct-RK3588-1.1.4')

        Returns:
            Dictionary with enhanced model metadata or None if not available
        """
        if not self.ollama_path or "/" not in self.ollama_path:
            raise ValueError("Invalid ollama path")

        if self._ollama_model_info:
            return self._ollama_model_info

        from core.model.storage_helpers.OllamaStorageHelper import OllamaStorageHelper

        ollama_model_info_path = OllamaStorageHelper.ollama_model_info_path(self)
        if ollama_model_info_path is None:
            return None

        try:
            if os.path.exists(ollama_model_info_path):
                self._ollama_model_info = OllamaModelInfo.load(ollama_model_info_path)
                return self._ollama_model_info
        except Exception as e:
            logger.exception(f"Error loading Ollama model info: {str(e)}")
            return None

        # else...
        try:
            formatted_model_name = self.ollama_path.replace(':', '/')
            url = f"https://ollama.com/api/registry/{formatted_model_name}/manifest"

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                ollama_data = response.json()

                logger.debug(f"Enhanced model info from OLLAMA API: {self.ollama_path}={ollama_data}")

                self._ollama_model_info = OllamaModelInfo(**ollama_data)
                self._ollama_model_info.save(OllamaStorageHelper.ollama_model_info_path(self))
                return self._ollama_model_info
            else:
                logger.debug(f"Failed to get OLLAMA data: {response.status_code}")
                return None
        except Exception as e:
            logger.exception(f"Error fetching OLLAMA model info: {str(e)}")
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
    param_pattern = re.search(
        PARAM_SIZE_PATTERN,
        content,
    )
    if param_pattern:
        int_size_value = None
        size_value = float(param_pattern.group(1))
        size_unit = param_pattern.group(2).lower()
        # Convert to billions if needed
        if size_unit == "b":
            int_size_value = int(size_value * 1_000_000_000)
        return size_value, size_unit, int_size_value
    else:
        return None, None, None
