import json
import os
import re
from enum import Enum

from pydantic_core import from_json
from typing import Optional, List, Tuple

from pydantic import BaseModel

from core.backends.backend import BACKEND_SUPPORTED_LIB_VERSION
from core.model.ModelInfo import ModelDetails
from core.model.ModelPath import ModelPath
from core.model.ModelType import ModelType
from core.model import logger
from core.model.converter.quantization_constants import RK_QUANT_FORMAT, ollama_quant_mapping
from core.model.models_constants import MODEL_SPECS, RK_TAGS_LIST


class ModelMetadataFormat(str, Enum):
    SIMPLE = "simple"
    BASIC = "basic"
    COMPLETE = "complete"

    @staticmethod
    def get_format(metadata_path: str):
        if not os.path.isfile(metadata_path):
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "conversion_date" in metadata:
                return ModelMetadataFormat.COMPLETE
            elif "model_id" in metadata:
                return ModelMetadataFormat.BASIC
            else:
                return ModelMetadataFormat.SIMPLE

METADATA_FILENAME="metadata.json"

class ModelMetadataParameters(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stop_sequences: List[str] = ["Human:", "Assistant:"]


class BasicModelMetadata(BaseModel):
    model_id: str
    quantization: str


class ModelMetadata(BasicModelMetadata):
    conversion_date: str
    parameters: ModelMetadataParameters

    def save(self, output_path: str) -> None:
        """
        Save model metadata to a JSON file.

        Args:
            metadata: The model metadata to save
            output_path: The directory to save the metadata in
        """
        try:
            # Save to JSON file
            metadata_path = os.path.join(output_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.model_dump_json(), f, indent=2)

            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    @classmethod
    def load(cls, metadata_path: str):
        """
        Load model metadata from a JSON file.

        Args:
            metadata_path: Path to the metadata JSON file

        Returns:
            The loaded model metadata
        """
        if ModelMetadataFormat.get_format(metadata_path) == ModelMetadataFormat.SIMPLE:
            raise ValueError("Metadata file is not in the correct format (cannot be SIMPLE).")

        try:
            with open(metadata_path, "r") as f:
                return ModelMetadata.model_validate(from_json(json.load(f), allow_partial=True))

        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise


class SimpleModelMetadata(BaseModel):
    """Metadata for a converted model."""
    name: str
    architecture: str
    quantization: str
    quantization_opt: Optional[int] = None
    quantization_hybrid_ratio: Optional[float] = None
    parameters: int
    context_length: int
    system_prompt: str
    temperature: float
    model_type: Optional[ModelType]


    @staticmethod
    def get_metadata_fields():
        return ["name", "architecture", "quantization", "parameters", "context_length", "system_prompt", "temperature", "model_type"]

    # model_name=Qwen3-1.7B-rk3588-1.2.1-unsloth-16k,
    @staticmethod
    def parse_model_name(model_name: str) -> Tuple[dict, dict, List[str]]:
        model_metadata = {}
        model_details = {}
        model_tags: List[str] = []
        splitted = model_name.split("-")
        start_pos = 0
        if MODEL_SPECS.get(splitted[0].lower()):
            name = splitted[0].lower()
            model_metadata.update({'name': name})
            model_tags.append(name)
            model_details.update({'model_family': name})
            start_pos = 1

        if start_pos == 0 and len(splitted) >= 2:
            if MODEL_SPECS.get(f"{splitted[0]}-{splitted[1]}".lower()):
                name = f"{splitted[0]}-{splitted[1]}".lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name})
                start_pos = 2
            elif MODEL_SPECS.get(f"{splitted[0]}_{splitted[1]}".lower()):
                name = f"{splitted[0]}_{splitted[1]}".lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name})
                start_pos = 2

        if len(splitted) > start_pos:
            if ModelPath.get_parameter_size(splitted[start_pos]):
                parameters = ModelPath.get_parameter_size(splitted[start_pos])
                model_metadata.update({'parameters': parameters})
                model_tags.append(parameters)
                model_details.update({'parameter_size': parameters})
                start_pos += 1

        if len(splitted) > start_pos:
            # rk3588-1.2.1
            for rk_tag in RK_TAGS_LIST:
                if rk_tag in splitted[start_pos]:
                    founded_rk_tag = rk_tag
                    model_tags.append(rk_tag)
                    model_details.update({'architecture': rk_tag})
                    start_pos += 1
                    if len(splitted) > start_pos+1:
                        lib_vers_match = re.search(r"(\d+\.\d+\.\d+)", splitted[start_pos+1])
                        if lib_vers_match:
                            lib_vers = lib_vers_match.group(1)
                            if lib_vers not in BACKEND_SUPPORTED_LIB_VERSION.get(founded_rk_tag, []):
                                raise ValueError(f"Library version {lib_vers} is not supported for {founded_rk_tag} backend.")
                            else:
                                model_tags.append(lib_vers)
                            start_pos += 1
                    break

        if len(splitted) > start_pos:
            # unsloth-16k
            if re.search(r"(\d+k)", splitted[start_pos]):
                context_length = re.search(r"(\d+k)", splitted[start_pos]).group(1)
                model_metadata.update({'context_length': context_length})
                model_details.update({'context_length': context_length})
                start_pos += 1
            elif 'unsloth' in splitted[start_pos]:
                model_tags.append('unsloth')
                start_pos += 1
                if re.search(r"(\d+k)", splitted[start_pos]):
                    context_length = re.search(r"(\d+k)", splitted[start_pos]).group(1)
                    model_metadata.update({'context_length': context_length})
                    model_details.update({'context_length': context_length})
                    start_pos += 1

        logger.debug(f"parse_model_name: Model metadata={model_metadata}")
        logger.debug(f"parse_model_name: Model details={model_details}")
        logger.debug(f"parse_model_name: Model tags={model_tags}")
        return model_metadata, model_details, model_tags

    # file=Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm,
    # repo=dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k
    @staticmethod
    def parse_file(file: str) -> Tuple[dict, dict, List[str]]:
        model_metadata = {}
        model_details = {}
        model_tags: List[str] = []
        splitted = file.split("-")
        start_pos = 0

        if MODEL_SPECS.get(splitted[0].lower()):
            name = splitted[0].lower()
            model_metadata.update({'name': name})
            model_tags.append(name)
            model_details.update({'model_family': name})
            start_pos = 1

        if start_pos == 0 and len(splitted) >= 2:
            if MODEL_SPECS.get(f"{splitted[0]}-{splitted[1]}".lower()):
                name = f"{splitted[0]}-{splitted[1]}".lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name})
                start_pos = 2
            elif MODEL_SPECS.get(f"{splitted[0]}_{splitted[1]}".lower()):
                name = f"{splitted[0]}_{splitted[1]}".lower()
                model_metadata.update({'name': name})
                model_tags.append(name)
                model_details.update({'model_family': name})
                start_pos = 2

        if len(splitted) > start_pos:
            if ModelPath.get_parameter_size(splitted[start_pos]):
                parameters = ModelPath.get_parameter_size(splitted[start_pos])
                model_metadata.update({'parameters': parameters})
                model_tags.append(parameters)
                model_details.update({'parameter_size': parameters})
                start_pos += 1

        if len(splitted) > start_pos:
            # rk3588
            for rk_tag in RK_TAGS_LIST:
                if rk_tag in splitted[start_pos]:
                    model_tags.append(rk_tag)
                    model_details.update({'architecture': rk_tag})
                    start_pos += 1
                    break

        if len(splitted) > start_pos:
            if splitted[start_pos].lower() in RK_QUANT_FORMAT:
                # w8a8-opt-0-hybrid-ratio-0.0.rkllm
                rk_quant_format = splitted[start_pos].lower()
                start_pos += 1
                if len(splitted) > start_pos:
                    if f"{splitted[start_pos-1]}_{splitted[start_pos]}".lower() in RK_QUANT_FORMAT:
                        rk_quant_format = f"{splitted[start_pos-1]}_{splitted[start_pos]}".lower()
                        start_pos += 1
                model_metadata.update({'quantization': rk_quant_format})
                model_tags.append(rk_quant_format)

                if len(splitted) > start_pos+1:
                    if "opt" == splitted[start_pos] and re.search(r"(\d+)", splitted[start_pos+1]):
                        quant_opt = re.search(r"(\d+)", splitted[start_pos+1]).group(1)
                        model_metadata.update({'quantization_opt': int(quant_opt)})
                        start_pos += 2
                if len(splitted) > start_pos+2:
                    if "hybrid" == splitted[start_pos] and "ratio" == splitted[start_pos+1] and re.search(r"(\d+\.\d+)", splitted[start_pos+2]):
                        quant_hybrid_ratio = re.search(r"(\d+\.\d+)", splitted[start_pos+2]).group(1)
                        model_metadata.update({'quantization_hybrid_ratio': float(quant_hybrid_ratio)})
                        start_pos += 2


        logger.debug(f"parse_file: Model metadata={model_metadata}")
        logger.debug(f"parse_file: Model details={model_details}")
        logger.debug(f"parse_file: Model tags={model_tags}")
        return model_metadata, model_details, model_tags

    @classmethod
    def compute(cls, model_path: ModelPath, model_details: ModelDetails, system_prompt: str) -> dict:
        model_metadata_from_name, model_details_from_name, model_tags_from_name = \
            SimpleModelMetadata.parse_model_name(model_path.model_name)

        model_metadata_from_file, model_details_from_file, model_tags_from_file = \
            SimpleModelMetadata.parse_file(model_path.endpoint_model_file)

        model_metadata_from_name.update(model_metadata_from_file)
        model_details_from_name.update(model_details_from_file)
        model_tags_from_name.extend(model_tags_from_file)

        to_return = {
            "name": model_metadata_from_name['name'],
            "architecture": model_metadata_from_name.get('architecture',
                                                         get_model_architecture(model_path.model_file)),
            "quantization": model_metadata_from_name.get('quantization',
                                                         ollama_quant_mapping.get(model_details.quantization_level)),
            "parameters": model_metadata_from_name.get('parameters',
                                                       model_details.parameter_size),
            "context_length": model_metadata_from_name.get('parameters',
                                                           model_details.context_length),
            "system_prompt": system_prompt,
            "temperature": model_details.temperature,
            "model_type": model_details.model_type
        }
        if 'quantization_opt' in model_metadata_from_name:
            to_return.update({'quantization_opt': model_metadata_from_name['quantization_opt']})
        if 'quantization_hybrid_ratio' in model_metadata_from_name:
            to_return.update({'quantization_hybrid_ratio': model_metadata_from_name['quantization_hybrid_ratio']})

        return to_return

    @classmethod
    def from_complete(cls, metadata: ModelMetadata):
        data: dict = {}
        for attr in metadata.__dict__:
            if attr == "model_id":
                data["name"] = metadata.__dict__[attr]
            elif attr == "parameters":
                for param_attr in metadata.parameters.__dict__:
                    data[param_attr] = metadata.parameters.__dict__[param_attr]

        return cls(**data)

    def save(self, output_path: str) -> None:
        """
        Save model metadata to a JSON file.

        Args:
            metadata: The model metadata to save
            output_path: The directory to save the metadata in
        """
        try:
            # Save to JSON file
            metadata_path = os.path.join(output_path, METADATA_FILENAME)
            with open(metadata_path, "w") as f:
                json.dump(self.model_dump_json(), f, indent=2)

            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    @classmethod
    def load(cls, metadata_path: str):
        """
        Load model metadata from a JSON file.

        Args:
            metadata_path: Path to the metadata JSON file

        Returns:
            The loaded model metadata
        """
        if ModelMetadataFormat.get_format(metadata_path) != ModelMetadataFormat.SIMPLE:
            raise ValueError(
                f"Metadata file is not in the correct format (search for SIMPLE, is {ModelMetadataFormat.get_format(metadata_path)}).")

        try:
            with open(metadata_path, "r") as f:
                return SimpleModelMetadata.model_validate(from_json(json.load(f), allow_partial=True))

        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise


def get_model_size(model_path: str) -> int:
    """Get the size of a model file in bytes."""
    return os.path.getsize(model_path)


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_model_architecture(model_path: str) -> Optional[str]:
    """Detect the model architecture from the model file."""
    # TODO: Implement architecture detection
    pass
