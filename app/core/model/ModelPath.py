import os
import re
from typing import Union

import requests

from core.config.config_utils import rkllama_config
from core.model.ModelName import ModelName, ModelType
from core.model import logger


class ModelPath(ModelName):
    huggingface_path: str
    endpoint_model_file: str
    _model_dir: Union[str|None] = None

    @property
    def model_dir(self):
        if not self._model_dir:
            self._model_dir = os.path.join(rkllama_config.paths.models, self.model_name.replace('.rkllm', ''))
        return self._model_dir

    @property
    def model_type(self) -> ModelType | None:
        for mtype in ModelType:
            if self.endpoint_model_file.endswith(mtype.get_extension()):
                return mtype
        return None


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
                    model_path + " " + (rkllama_config.config_utils.get("description") or ""),
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
            if "sibling_models" in data:
                for sibling in core.rkllama_config.config_utils.get("sibling_models", []):
                    if core.rkllama_config.config_utils.get("rfilename", "").endswith(".rkllm"):
                        data["has_rkllm"] = True
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
        debug_mode = core.rkllama_config.config_utils.is_debug_mode()
        if debug_mode:
            logger.exception(f"Error fetching HF model info: {str(e)}")
        return None


def GetModels():
    print("Retrieving models...")

    MODEL_PATH = os.path.join(core.rkllama_config.config_utils.get_path("models"))

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
