import json
import os
from enum import Enum

from pydantic_core import from_json
from typing import Optional, List

from pydantic import BaseModel

from core.model.ModelType import ModelType
from core.model import logger

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
    parameters: int
    context_length: int
    system_prompt: str
    temperature: float
    model_type: Optional[ModelType]

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
