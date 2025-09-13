import json
import os
from pydantic_core import from_json
from typing import Optional

from pydantic import BaseModel

from core.model.ModelName import ModelType


class ModelMetadata(BaseModel):
    """Metadata for a converted model."""
    name: str
    architecture: str
    quantization: str
    parameters: int
    context_length: int
    system_prompt: str
    temperature: float
    model_type: Optional[ModelType]


def save_model_metadata(metadata: ModelMetadata, output_path: str) -> None:
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
            json.dump(metadata.model_dump_json(), f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        raise


def load_model_metadata(metadata_path: str) -> ModelMetadata:
    """
    Load model metadata from a JSON file.

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        The loaded model metadata
    """
    try:
        with open(metadata_path, "r") as f:
            return ModelMetadata.model_validate(from_json(json.load(f), allow_partial=True))

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
