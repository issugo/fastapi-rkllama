from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from core.model.ModelInfo import ModelDetails
from core.model.ModelPath import ModelPath

MODELFILE_CONFIG_NAME = ".config"


class ModelFileInfo(ModelPath):
    system_prompt: Optional[str] = Field(default=None, description="System prompt")

    _simple_model_metadata = None

    @property
    def simple_model_metadata(self) -> Any:
        from core.model.ModelMetadata import SimpleModelMetadata

        if self._simple_model_metadata:
            return self._simple_model_metadata

        # compute metadata from endpoint_model_file name using ModelPath.extract_model_details
        data = SimpleModelMetadata.compute(
            model_path=self,
            model_details=ModelDetails.from_model_path(model_path=self),
            system_prompt=self.system_prompt,
        )
        return SimpleModelMetadata(**data)

    @simple_model_metadata.setter
    def simple_model_metadata(self, value):
        self._simple_model_metadata = value

    def modelfile_config_path_using_model_dir_path(self, model_dir_path: Path) -> Path:
        return model_dir_path / f"{self.endpoint_model_file}{MODELFILE_CONFIG_NAME}"

    @property
    def modelfile_config_path(self) -> Path:
        return self.modelfile_config_path_using_model_dir_path(self.model_dir_path)
