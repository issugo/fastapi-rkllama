from os import stat_result
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from core.config.PathsConfig import PATH_KEY
from core.config.config_utils import get_settings
from core.model import logger
from core.model.HfFileInfo import HfFileInfo
from core.model.ModelInfo import ModelInfo
from core.model.models_constants import validate_model_id
from core.model.storage_helpers.SupplierFileInfo import Supplier
from core.model.suppliers_model_info import OllamaModelInfo, HFModelInfo
from core.model.ModelMetadata import ModelMetadataFormat, BasicModelMetadata, SimpleModelMetadata, ModelMetadata, \
    ModelMetadataNotFoundException
from core.model.ModelPath import ModelPath, ModelDirError, ModelDirException, ModelNotFoundException, ModelException
from core.model.OllamaManifest import OllamaManifest


class ModelSharedData(BaseModel):
    global_status: int = -1
    global_text: List[str] = []


class Model(BaseModel):
    id: str
    st_atime: float
    st_mtime: float
    st_ctime: float
    size: int
    digest: str
    model_path: ModelPath
    # model_info contains only model file stats, and nothing in relation with model content configuration
    model_info: ModelInfo
    # model_metadata contains model configuration
    model_metadata: Optional[BasicModelMetadata|SimpleModelMetadata|ModelMetadata] = Field(default=None, description="Model metadata")

    _supplier: Optional[Supplier] = None
    _supplier_model_info: Optional[OllamaModelInfo|HFModelInfo] = None

    #model_file: ModelFile # TODO model_file has a model, not the opposite
    #backend: Optional[Backend]
    #shared_data: ModelSharedData
    #usage_lock: threading.Lock

    @property
    def supplier_model_info(self) -> OllamaModelInfo|HFModelInfo|None:
        return self._supplier_model_info

    def get_metadata_format(self) -> ModelMetadataFormat | None:
        if self.model_metadata is None:
            return None
        return self.model_metadata.get_format()

    @classmethod
    def from_model_path(cls, model_path: ModelPath) -> Any:
        if model_path.model_exists:
            model_stat: stat_result = model_path.endpoint_model_file_path.resolve().stat()
            model_info = None
            digest = None
            size = None
            supplier_model_info = None
            try:
                # when the supplier is Ollama, file_info is an OllamaManifest
                # when the supplier is Ollama, model_file_info is an OllamaModelInfo
                if model_path.ollama_file_info_exists:
                    if model_path.ollama_model_info_exists:
                        supplier_file_info: OllamaManifest = model_path.ollama_file_info
                        size = supplier_file_info.size
                        digest = supplier_file_info.digest
                        supplier_model_info: OllamaModelInfo = model_path.ollama_model_info
                        model_info: ModelInfo = ModelInfo.from_ollama_model_info(supplier_model_info, model_path, size, digest, model_stat)
            except ValueError as e:
                logger.error(f"Error loading ModelFile: {str(e)}", exc_info=True)
            try:
                # when the supplier is Rkllama, file_info is a HfFileInfo
                # when the supplier is Rkllama, model_file_info is a HFModelInfo
                if model_path.huggingface_file_info_exists:
                    if model_path.huggingface_model_info_exists:
                        supplier_file_info: HfFileInfo = model_path.huggingface_file_info
                        size = supplier_file_info.size
                        digest = supplier_file_info.digest
                        supplier_model_info: HFModelInfo = model_path.huggingface_model_info
                        model_info: ModelInfo = ModelInfo.from_hf_model_info(supplier_model_info, model_path, size, digest, model_stat)
            except ValueError as e:
                logger.error(f"Error loading ModelFile: {str(e)}", exc_info=True)
            if model_info is not None:
                model: Model = Model(
                    id=validate_model_id(model_path.model_id),
                    st_atime=model_stat.st_atime,
                    st_mtime=model_stat.st_mtime,
                    st_ctime=model_stat.st_ctime,
                    size=size,
                    digest=digest,
                    model_path=model_path,
                    model_info=model_info,
                )
                model._supplier = supplier_model_info.supplier
                model._supplier_model_info = supplier_model_info
            else:
                raise ModelNotFoundException(model_path.model_id)
            return model
        else:
            raise ModelNotFoundException(model_path.model_id)

    @classmethod
    def clean_metadata(cls, model_path: ModelPath):
        logger.debug(f"ModelFile.clean_metadata(model_path={model_path})")

    @classmethod
    def clean(cls, model_path: ModelPath):
        logger.debug(f"Model.clean(model_path={model_path})")
        cls.clean_metadata(model_path)

    def load_metadata(self) -> Any:
        logger.debug(f"ModelFile.load_metadata()")
        try:
            model_metadata: SimpleModelMetadata = SimpleModelMetadata.load(model_path=self.model_path)
            return model_metadata
        except ModelMetadataNotFoundException as e:
            error_msg = f"Error loading model metadata: {str(e)}"
            logger.error(f"Model.load_metadata(): {error_msg}", exc_info=True)
            raise e

    @classmethod
    def load(cls, model_path: ModelPath) -> Any:
        logger.debug(f"Model.load(model_path={model_path})")
        try:
            model: Model = cls.from_model_path(model_path)
            model.model_metadata = model.load_metadata()
            return model
        except ModelException as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(f"Model.load(): {error_msg}", exc_info=True)
            raise e


    def save_metadata(self):
        logger.debug(f"self.save_metadata()")
        self.model_metadata.save(model_path=self.model_path)

    def save(self):
        logger.debug(f"Model.save(model_path={self.model_path})")
        self.save_metadata()
        # not saving model, only metadata (a model is pulled or converted only)
        # TODO: catch Exception then clean all

    @classmethod
    def list(cls) -> List[Any]:
        models_dir = Path(get_settings().get_path(PATH_KEY.MODELS))

        if not models_dir.exists():
            raise ModelDirException(ModelDirError.NOT_EXIST)
        elif not models_dir.is_dir():
            raise ModelDirException(ModelDirError.INVALID)
        else:
            models = []
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    for dir_content in model_dir.iterdir():
                        if dir_content.is_symlink():
                            try:
                                model_stat = dir_content.resolve().stat()
                                model_path: ModelPath = ModelPath(model_name=model_dir.name,
                                                                  endpoint_model_file=dir_content.name,
                                                                  endpoint_model_file_size=model_stat.st_size,
                                                                  )
                                model = Model.load(model_path)
                                models.append(model)
                            except Exception as e:
                                logger.error(f"Error loading ModelFile: {e}", exc_info=True)
                                continue

            return models


    """
    def __init__(self, model_file: ModelFile, /, **data: Any):
        super().__init__(**data)

        if not os.path.exists(model_file.model_dir):
            err_msg = f"Model directory '{model_file.model_name}' not found."
            logger.error(f"Error loading model: {err_msg}", exc_info=True)
            raise ModelException(err_msg)

        if not os.path.exists(os.path.join(model_file.model_dir, "Modelfile")) and (
            model_file.huggingface_path is None and model_file.From is None
        ):
            err_msg = f"Modelfile not found in '{model_file.model_name}' directory."
            logger.error(f"Error loading model: {err_msg}", exc_info=True)
            raise ModelException(err_msg)
        elif model_file.huggingface_path is not None and model_file.From is not None:
            ModelFile.create_modelfile(model_file)
            time.sleep(0.1)

        endpoint_model_file = model_file.endpoint_model_file
        huggingface_path = model_file.huggingface_path
        try:
            # Load modelfile
            load_dotenv(model_file.file, override=True)

            endpoint_model_file = os.getenv("FROM", endpoint_model_file)
            huggingface_path = os.getenv("HUGGINGFACE_PATH", huggingface_path)
        except RuntimeError as e:
            logger.error(f"Error loading modelfile: {e}", exc_info=True)
            raise ModelException(e)

        # View config Vars
        logger.info(f"FROM: {endpoint_model_file}\nHuggingFace Path: {huggingface_path}")

        if not endpoint_model_file or not huggingface_path:
            err_msg = "FROM or HUGGINGFACE_PATH not defined in Modelfile."
            logger.error(f"Error loading model: {err_msg}", exc_info=True)
            raise ModelException(err_msg)

        # Get model parameters if not provided
        if not model_file.options:
            model_file.options = model_file.full_options

        try:
            from core.backends.GlobalState import GLOBAL_STATE
            from core.backends.rkllm.rkllm_backend import RKLLMBackend

            # Change value of model_id with huggingface_path
            GLOBAL_STATE.loaded_model_hfpath = huggingface_path
            GLOBAL_STATE.backend = RKLLMBackend(
                model_path=os.path.join(model_file.model_dir, endpoint_model_file),
                model_dir=self.model_dir,
                options=self.options
            )
        except RuntimeError as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise ModelException(e)

        self.model_file = model_file

        self.shared_data = ModelSharedData()
        self.usage_lock = threading.Lock()  # old verrou
    """

    def unload(self):
        if self.backend:
            self.backend.release()
            self.backend = None
        self.model_file = None
