import os
import re
from typing import Union

import requests

from core.config.config_utils import rkllama_config
from core.model.ModelInfo import ModelDetails
from core.model.ModelName import ModelName
from core.model.ModelType import ModelType
from core.model import logger
from core.model.converter.quantization_constants import quant_patterns, quant_mapping, ollama_quant_mapping


class ModelPath(ModelName):
    huggingface_path: str
    endpoint_model_file: str
    _model_dir: Union[str|None] = None

    @property
    def model_dir(self):
        if not self._model_dir:
            model_ext = self.model_type.get_extension()
            if model_ext:
                default_relative_dir = self.model_name.replace(model_ext, '')
            else:
                default_relative_dir = self.model_name
            self._model_dir = os.path.join(rkllama_config.paths.models, default_relative_dir)
        return self._model_dir

    @property
    def model_type(self) -> ModelType | None:
        for mtype in ModelType:
            if self.endpoint_model_file.endswith(mtype.get_extension()):
                return mtype
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

    @staticmethod
    def get_parameter_size(model_name: str)-> str | None:
        # Extract parameter size (e.g., 3B, 7B, 13B)
        param_size_match = re.search(r"(\d+\.?\d*)(b|B)", model_name)
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

    def extract_model_details(self) -> ModelDetails:
        """
        Extract model parameter size and quantization type from model name

        Args:
            model_name: Model name or file path

        Returns:
            Dictionary with parameter_size and quantization_level
        """
        # Initialize default values
        details = ModelDetails(parameter_size="Unknown", quantization_level="Unknown")

        # Remove path and extension if present
        if isinstance(self.model_name, str):
            basename = os.path.basename(self.model_name).replace(self.model_type.get_extension(), "")
        else:
            basename = str(model_name)

        parameter_size = ModelPath.get_parameter_size(basename)
        if parameter_size:
            details["parameter_size"] = parameter_size

        for quant_type, pattern in quant_patterns:
            if re.search(pattern, basename, re.IGNORECASE):
                # Use Ollama-style quantization name if available
                details["quantization_level"] = quant_mapping.get(quant_type)
                break

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


def get_huggingface_model_info(model_path):
    """
    Fetch model metadata from Hugging Face API if available.

    Args:
        model_path: HuggingFace repository path (e.g., 'c01zaut/Qwen2.5-3B-Instruct-RK3588-1.1.4')

    Returns:
        Dictionary with enhanced model metadata or None if not available
    """
    try:
        if not model_path or "/" not in model_path:
            return None

        # Get DEBUG_MODE from configuration
        debug_mode = rkllama_config.is_debug_mode()

        # Extract repo_id from HUGGINGFACE_PATH
        url = f"https://huggingface.co/api/models/{model_path}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()

            # Process and enhance the metadata
            if "tags" not in data:
                data["tags"] = []

            # Extract additional info from readme if available
            if "cardData" not in data:
                data["cardData"] = {}

            # Try to extract parameter size from model name if not in cardData
            if "params" not in data["cardData"]:
                # Look for patterns like "7b", "3B", "1.5B" in model name or description
                param_pattern = re.search(
                    r"(\d+\.?\d*)([bB])",
                    model_path + " " + (data.get("description") or ""),
                )
                if param_pattern:
                    size_value = float(param_pattern.group(1))
                    size_unit = param_pattern.group(2).lower()
                    # Convert to billions if needed
                    if size_unit == "b":
                        data["cardData"]["params"] = int(size_value * 1_000_000_000)

            # Extract important information from the description
            description = rkllama_config.config_utils.get("description", "")
            if description:
                # Look for model details in the description
                quant_pattern = re.search(
                    r"([qQ]\d+_\d+|int4|int8|fp16|4bit|8bit)", description
                )
                if quant_pattern:
                    data["quantization"] = quant_pattern.group(1)

                # Check for mentions of specific architectures
                architectures = {
                    "llama": "llama",
                    "mistral": "mistral",
                    "qwen": "qwen",
                    "deepseek": "deepseek",
                    "phi": "phi",
                    "gemma": "gemma",
                    "baichuan": "baichuan",
                    "yi": "yi",
                }

                for arch_name, arch_value in architectures.items():
                    if arch_name.lower() in description.lower():
                        data["architecture"] = arch_value
                        if arch_name.lower() not in data["tags"]:
                            data["tags"].append(arch_name.lower())

            # Try to extract language information
            languages = []
            language_patterns = {
                "english": "en",
                "chinese": "zh",
                "multilingual": None,  # Special case
                "french": "fr",
                "german": "de",
                "spanish": "es",
                "japanese": "ja",
            }

            for lang_name, lang_code in language_patterns.items():
                if (
                    lang_name.lower() in description.lower()
                    or lang_name.lower() in " ".join(data["tags"]).lower()
                ):
                    if lang_name == "multilingual":
                        # For multilingual models, add common languages
                        languages.extend(["en", "zh", "fr", "de", "es", "ja"])
                    elif lang_code and lang_code not in languages:
                        languages.append(lang_code)

            # If we found languages, add them
            if languages:
                data["languages"] = list(set(languages))  # Remove duplicates
            elif "en" not in rkllama_config.config_utils.get("languages", []):
                # Default to English if no languages detected
                data["languages"] = ["en"]

            # Add RK tags if they exist
            rk_patterns = ["rk3588", "rk3576", "rkllm", "rockchip"]
            for pattern in rk_patterns:
                if (
                    pattern in model_path.lower()
                    or pattern in " ".join(data["tags"]).lower()
                    or pattern in description.lower()
                ):
                    if "rockchip" not in data["tags"]:
                        data["tags"].append("rockchip")
                    if pattern not in data["tags"] and pattern != "rockchip":
                        data["tags"].append(pattern)

            # Add metadata about model capabilities
            if 'sibling_models' in data:
                for sibling in data.get('sibling_models', []):
                    if sibling.get('rfilename', '').endswith('.rkllm'):
                        data['has_rkllm'] = True
                        break

            # Extract license information
            if "license" in data and data["license"]:
                # Map HF license IDs to human-readable names
                license_mapping = {
                    "apache-2.0": "Apache 2.0",
                    "mit": "MIT",
                    "cc-by-4.0": "Creative Commons Attribution 4.0",
                    "cc-by-sa-4.0": "Creative Commons Attribution-ShareAlike 4.0",
                    "cc-by-nc-4.0": "Creative Commons Attribution-NonCommercial 4.0",
                    "cc-by-nc-sa-4.0": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0",
                }

                license_id = data["license"].lower()
                data["license_name"] = license_mapping.get(license_id, data["license"])
                data["license_url"] = (
                    f"https://huggingface.co/{model_path}/blob/main/LICENSE"
                )

            if debug_mode:
                logger.debug(f"Enhanced model info from HF API: {model_path}")

            return data
        else:
            if debug_mode:
                logger.debug(f"Failed to get HF data: {response.status_code}")
            return None
    except Exception as e:
        debug_mode = rkllama_config.is_debug_mode()
        if debug_mode:
            logger.exception(f"Error fetching HF model info: {str(e)}")
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


MODELFILE_NAME:str = "Modelfile"
