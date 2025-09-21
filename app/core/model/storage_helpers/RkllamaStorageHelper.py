from core.model.ModelFile import ModelFile
from core.model.storage_helpers.OllamaModelStorageHelper import OllamaModelStorageHelper
from core.model.storage_helpers.StorageHelper import StorageHelper


class RkllamaStorageHelper(StorageHelper):
    ollama_model_storage_helper: OllamaModelStorageHelper
    model_file: ModelFile

    def store(self):

        if self.model_file._huggingface_model_info:
            # store HF infos
            self.model_file._huggingface_model_info.save(self.model_file.model_dir)

        # search for metadata file in '.'+model_name directory, if not found, create one by dumping the model_file.simple_model_metadata object into the directory
        metadata_path = os.path.join(dotdir, METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            model_file.simple_model_metadata.save(dotdir)

        raise NotImplementedError