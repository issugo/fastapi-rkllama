import os
from pathlib import Path
from typing import Optional, Any, Union

from pydantic import BaseModel, Field

from core.config.PathsConfig import PATH_KEY
from core.config.warnings import deprecated
from core.model import logger
from core.model.ModelType import ModelType
from core.config.config_utils import get_settings
from core.model.models_constants import validate_model_id


class ModelNameException(Exception):
    pass


class ModelName(BaseModel):
    model_name: str
    model_format: Optional[ModelType] = Field(default=None, alias="model_type")

    _model_dir: Union[str | None] = None

    @staticmethod
    def model_id_to_path(model_id: str, author: str = None) -> str:
        model_id = validate_model_id(model_id)
        if author:
            model_id = os.path.join(
                author,
            )
        return model_id.replace(":", "/")

    @classmethod
    def from_model_id(cls, model_id: str) -> Any:
        model_name: str = model_id.split(":")[0].split("/")[0]
        if model_name == model_id:
            raise ModelNameException(f"Invalid model id: {model_id}")
        return cls(model_name=model_name)

    @property
    def model_type(self) -> ModelType | None:
        return self.model_format

    @model_type.setter
    def model_type(self, value: ModelType):
        self.model_format = value

    @staticmethod
    def model_dir_using_model_name(model_name: str) -> str:
        return os.path.join(get_settings().paths.models, model_name)

    @property
    def model_dir(self) -> str:
        if not self._model_dir:
            if self.model_format:
                logger.debug(
                    f"Using default relative dir for model {self.model_name} with type {self.model_format}"
                )
                default_relative_dir = self.model_name.replace(
                    self.model_format.get_extension(), ""
                )
            else:
                default_relative_dir = self.model_name
            self._model_dir = ModelName.model_dir_using_model_name(
                model_name=default_relative_dir
            )
        return self._model_dir

    @property
    def model_dir_path(self) -> Path:
        return Path(self.model_dir)

    def endpoint_model_file_path_with_endpoint(self, endpoint_model_file: str) -> Path:
        return self.model_dir_path / endpoint_model_file


@deprecated("use Model.size instead.")
def get_model_size(model_name) -> int:
    """
    Get the size of a model
    Args:
        model_name: The name of the model directory
    Returns:
        The size of the model in bytes or None if not found
    """

    # Get the models directory
    models_dir: Path = Path(get_settings().get_path(PATH_KEY.MODELS))
    model_path: Path = models_dir / model_name

    # check for the RKLLM file to get his size
    if model_path.is_dir():
        for file in os.listdir(model_path):
            if file.endswith(".rkllm"):
                size = os.path.getsize(os.path.join(model_path, file))
                return size

    return None
