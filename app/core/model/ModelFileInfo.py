from pathlib import Path
from typing import Any

from core.model.ModelPath import ModelPath

MODELFILE_CONFIG_NAME = ".config"
MODELFILE_METADATA_NAME = ".metadata"


class ModelFileInfo(ModelPath):
    system_prompt: str = ""

    _simple_model_metadata = None

    @property
    def simple_model_metadata(self) -> Any:
        from core.model.ModelMetadata import SimpleModelMetadata
        if self._simple_model_metadata:
            return self._simple_model_metadata

        # compute metadata from endpoint_model_file name using ModelPath.extract_model_details
        data = SimpleModelMetadata.compute(
            model_path=self,
            model_details=self.extract_model_details(),
            system_prompt=self.system_prompt)
        return SimpleModelMetadata(**data)

    @simple_model_metadata.setter
    def simple_model_metadata(self, value):
        self._simple_model_metadata = value

    @property
    def modelfile_config_path(self) -> Path:
        return self.model_dir_path / f"{self.endpoint_model_file}{MODELFILE_CONFIG_NAME}"

    @property
    def modelfile_metadata_path(self) -> Path:
        return self.model_dir_path / f"{self.endpoint_model_file}{MODELFILE_METADATA_NAME}"
