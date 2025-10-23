import json
import re
from typing import Tuple, Any
import requests

from core.model.storage_helpers import logger

from core.model.ModelPath import int_parameters_size
from core.model.models_constants import MODEL_ARCHITECTURES, LANGUAGE_PATTERNS, LANGUAGE_MULTILINGUAL_LIST, \
    LANGUAGE_DEFAULT, RK_TAGS_LIST, LICENSE_NAME_MAPPING


class HuggingfaceFileSystem:
    @staticmethod
    def load_model_info(huggingface_path : str):
        try:
        # Extract repo_id from HUGGINGFACE_PATH
            url = f"https://huggingface.co/api/models/{huggingface_path}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                hf_data = response.json()
                logger.debug(f"load_model_info(): HF data={hf_data}")

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
                        int_parameters_size(content=huggingface_path + " " + (hf_data.get("description") or ""))
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
                            pattern in huggingface_path.lower()
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
                        f"https://huggingface.co/{huggingface_path}/blob/main/LICENSE"
                    )

                logger.debug(f"Enhanced model info from HF API: {huggingface_path}={hf_data}")

                return hf_data
            else:
                err_msg = f"Failed to get HF data: {response.status_code}"
                logger.error(err_msg)
                raise Exception(err_msg)
        except Exception as e:
            logger.exception(f"Error fetching HF model info: {str(e)}")
            raise e
