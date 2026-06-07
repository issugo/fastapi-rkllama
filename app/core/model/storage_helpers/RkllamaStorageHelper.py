from pathlib import Path

from core.model.ModelFile import ModelFile
from core.model.ModelFileInfo import ModelFileInfo
from core.model.ModelPath import ModelPath
from core.model.storage_helpers.StorageHelper import StorageHelper

HF_FILEINFO_EXTENSION = ".HfFileInfo"
HF_MODELINFO_FILENAME = "ModelInfo.json"


class RkllamaStorageHelper(StorageHelper):
    model_file: ModelFile

    def __init__(self, ollama_model_storage_helper, model_file: ModelFile):
        super().__init__(ollama_model_storage_helper)
        self.model_file = model_file
        self.logger.debug(
            f"RkllamaStorageHelper: model_file.huggingface_model_info_exists={model_file.huggingface_model_info_exists}"
        )
        self.logger.debug(
            f"RkllamaStorageHelper: model_file.huggingface_file_info_exists={model_file.huggingface_file_info_exists}"
        )

    @staticmethod
    def huggingface_model_info_path_using_model_dir(model_dir: str) -> Path:
        return Path(model_dir) / HF_MODELINFO_FILENAME

    @staticmethod
    def huggingface_model_info_path(model_path: ModelPath) -> Path:
        return RkllamaStorageHelper.huggingface_model_info_path_using_model_dir(
            model_dir=model_path.model_dir
        )

    @staticmethod
    def huggingface_file_info_path(model_path: ModelPath) -> Path:
        """where to store the HfFileInfo object in the model directory"""
        return Path(
            RkllamaStorageHelper.huggingface_file_info_path_from_raw(
                model_name=model_path.model_name,
                endpoint_model_file=model_path.endpoint_model_file,
            )
        )

    @staticmethod
    def huggingface_file_info_path_from_raw(
        model_name: str, endpoint_model_file: str
    ) -> str:
        """where to store the HfFileInfo object in the model directory"""
        return f"{ModelPath.model_dir_using_model_name(model_name)}/{endpoint_model_file}{HF_FILEINFO_EXTENSION}"

    @property
    def model_link(self) -> Path:
        return (
            Path(
                ModelPath.model_dir_using_model_name(
                    self.model_file.model_file_info.model_name
                )
            )
            / self.model_file.model_file_info.endpoint_model_file
        )

    def clean(
        self, generic_model_file: ModelFile, generic_model_file_info: ModelFileInfo
    ):
        try:
            model_link = Path(self.model_link)
            if model_link.exists():
                model_link.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning model link: {str(e)}")

        try:
            huggingface_file_info_file_path = (
                RkllamaStorageHelper.huggingface_file_info_path(
                    self.model_file.model_file_info
                )
            )
            if huggingface_file_info_file_path.exists():
                huggingface_file_info_file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning huggingface_file_info: {str(e)}")

        try:
            huggingface_model_info_file_path = (
                RkllamaStorageHelper.huggingface_model_info_path(
                    self.model_file.model_file_info
                )
            )
            if huggingface_model_info_file_path.exists():
                huggingface_model_info_file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning huggingface_model_info: {str(e)}")

    def store(self):
        if self.model_file.huggingface_model_info_exists:
            file_path = RkllamaStorageHelper.huggingface_model_info_path(
                self.model_file.model_file_info
            )
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True)
            self.model_file.huggingface_model_info.save(file_path=file_path)

        if self.model_file.huggingface_file_info_exists:
            file_path = RkllamaStorageHelper.huggingface_file_info_path(
                self.model_file.model_file_info
            )
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True)
            self.model_file.huggingface_file_info.save(file_path=file_path)

        self._store_model_link()
