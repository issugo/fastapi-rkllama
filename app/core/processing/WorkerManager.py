import logging
import threading
import time
from typing import List
from multiprocessing import Process

import psutil
from datetime import datetime, timedelta
from operator import attrgetter

from core.backends.backend import BackendType
from core.backends.rkllm.classes import RKLLMInferMode, RKLLMInputType
from core.config.RKLLAMAConfig import RKLLAMASettings
from core.model.Model import ModelSharedData
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.model.ModelPath import ModelPath
from core.processing.BaseDomainId import BaseDomainId
from core.processing.tasks.Tasks import Tasks
from core.processing.tasks import (
    InferenceTask,
    EmbeddingTask,
    VisionEncoderTask,
    AbortInferenceTask,
    UnloadModelTask,
    ClearCacheTask,
)
from core.processing.workers import RkllmWorker, RknnWorker
from core.processing.workers.Worker import Worker

settings: RKLLAMASettings | None = None
DEBUG_MODE: bool | None = None

npu_lock: threading.Lock = threading.Lock()

logger = logging.getLogger("rkllama.worker_manager")


# Class to manage the workers for RKLLM models
class WorkerManager:
    def __init__(self, backend_type: BackendType):
        global settings
        if settings is None:
            from core.config import config_utils

            settings = config_utils.get_settings()

        global DEBUG_MODE
        if DEBUG_MODE is None:
            DEBUG_MODE = settings.is_debug_mode()

        if DEBUG_MODE:
            logger.debug("start WorkerManager")

        self.workers = {}  # (name -> Worker)
        self.backend_type: BackendType = backend_type
        self.model_shared_data: ModelSharedData = ModelSharedData()

    @classmethod
    def lock(cls):
        npu_lock.acquire_lock()

    @classmethod
    def unlock(cls):
        npu_lock.release_lock()

    def unload_expired_models(self) -> int | None:
        """
        Unload/stop workers for expired models
        """
        # Get all expired models
        expired_models = [
            model
            for model in self.workers.keys()
            if datetime.now() > self.workers[model].worker_model_info.expires_at
        ]

        # Unload/stop the expired modelfile
        for model_name in expired_models:
            logger.info(f"Detected expired modelfile: {model_name}")
            self.stop_worker(model_name)

    def get_available_base_domain_id(self, reverse_order=False) -> BaseDomainId | None:
        """
        Returns the smallest available integer between 1 and 10
        that is not already used as 'base_domain_id' in the current list of worker process.
        If all numbers from 1 to 10 are taken, returns None.

        Args:
            reverse_order (bool): If true, search from the highest to the lowest.

        Returns:
            int | None: The available base_domain_id or None if all are taken.
        """
        # Get all used base domain ids
        used_base_domain_ids = [
            self.workers[model].worker_model_info.base_domain_id
            for model in self.workers.keys()
        ]

        # Get the max id of a domain base:
        max_domain_id = (
            settings.server.max_number_models_loaded_in_memory
        )  # config.get("modelfile", "max_number_models_loaded_in_memory")

        if reverse_order:
            # CHeck fir available from the highest to the lowest
            candidates_range = range(max_domain_id, 0, -1)
        else:
            # CHeck first available from the lowest to the highest
            candidates_range = range(1, max_domain_id + 1)

        # CHeck fir available
        for candidate in candidates_range:
            if candidate not in used_base_domain_ids:
                return candidate
        return None

    def exists_model_loaded(self, model_id: str) -> bool:
        """
        Check if a modelfile with the given modelfile exists in the dict of workers
        Args:
            model_id (str): Model name to check if already loaded in memory.

        """
        return model_id in self.workers.keys()

    # TODO: add lora_model_path=None in model
    def add_worker(
        self, modelfile: ModelFile, full_model_parameters: FullModelParameters
    ) -> tuple[Worker, Process]:
        """
        Add a process worker to run inferences call from a specific modelfile

        Args:
            model_name (str): modelfile name to load in memory
        """
        model_path: ModelPath = modelfile.model.model_path
        if model_path.model_id not in self.workers.keys():
            # Get the available domain id for the RKLLM process
            base_domain_id: BaseDomainId = self.get_available_base_domain_id()

            # Add the worker to the dictionary of workers
            match self.backend_type:
                case BackendType.RKLLM:
                    worker_model: Worker = RkllmWorker(
                        modelfile, full_model_parameters, base_domain_id
                    )
                case BackendType.RKNN:
                    worker_model: Worker = RknnWorker(
                        modelfile, full_model_parameters, base_domain_id
                    )
                case _:
                    raise Exception(f"unsupported backend type {self.backend_type}")

            # Check if available memory in server
            if not self.is_memory_available_for_model(
                worker_model.worker_model_info.size
            ):
                # Unload the oldest modelfile until memory avilable
                self.unload_oldest_models_from_memory(
                    worker_model.worker_model_info.size
                )

            # Initialize of worker/modelfile
            model_loaded = worker_model.create_worker_process()

            # Check the load of the modelfile
            if not model_loaded:
                # Error loading the modelfile
                return None, None
            else:
                # Add the worker to the dictionary of workers
                self.workers[model_path.model_id] = worker_model
                logger.info(
                    f"Worker for modelfile {model_path.model_id} created and running..."
                )
                return worker_model, model_loaded
        else:
            logger.info(f"Worker for modelfile {model_path.model_id} reused...")
            worker_model: Worker = self.workers.get(model_path.model_id)
            return worker_model, worker_model.process

    def unload_oldest_models_from_memory(self, memory_required):
        """
        Unload the oldest models from memory
        Args:
            memory_required (int) -> Size of memory need by the modelfile to load
        """
        # From the dictionary of workers, we create an array of worker info that holds the size of each one
        worker_models_info = [
            self.workers[model].worker_model_info for model in self.workers.keys()
        ]

        # Loop over the array by the oldest worker modelfile
        for worker_model_info in sorted(
            worker_models_info, key=attrgetter("last_call")
        ):
            logger.info(
                f"Unloading modelfile {worker_model_info.modelfile} to gain free memory (at least {memory_required})"
            )
            # Stop the first oldest modelin memory
            self.stop_worker(worker_model_info.model)

            # Wait a second to refresh memory system
            time.sleep(1)

            # CHeck if now memory available for the new modelfile to load
            if self.is_memory_available_for_model(memory_required):
                break

    def is_memory_available_for_model(self, model_size) -> bool:
        """
        Check if exist memory available for modelfile load
        Args:
            model_size (int) -> Size of the modelfile to load
        """
        return (
            psutil.virtual_memory().available + psutil.virtual_memory().free
        ) > model_size

    def send_task(self, model_id: str, task):
        """
        Send a task to execute for the RKLLM modelfile
        Args:
            model_id (str): Worker name to send the task.
            task (tuple (name_task,args)): Task to send to the worker

        """
        if model_id in self.workers:
            # Send the TASK to the modelfile with the communication queue of the modelfile
            self.workers[model_id].task_q.put(task)

            # Update the worker modelfile info with the invocation
            self.workers[model_id].worker_model_info.last_call = datetime.now()
            self.workers[
                model_id
            ].worker_model_info.expires_at = datetime.now() + timedelta(
                minutes=settings.server.max_minutes_loaded_in_memory
            )

    def get_result(self, model_id: str):
        """
        Get the result of a task executed for the RKLLM modelfile

        Args:
            model_id (str): Worker name to get the response.

        Returns:
            Queue: Queue for the worker where the response is stored.
        """
        if model_id in self.workers:
            # Get the queue of the responses of the worker
            return self.workers[model_id].result_q
        return None

    def unload_model(self, model_id: str):
        return self.stop_worker(model_id)

    def stop_worker(self, model_id: str):
        """
        Stop/Unload a modelfile worker

        Args:
            model_name (str): Workers to unload.

        """
        if model_id in self.workers.keys():
            # Get the queue of tasks of the worker

            # Send the abort task of the modelfile if currently is running some inference
            self.workers[model_id].task_q.put(AbortInferenceTask())

            # Send the unload task of the modelfile
            self.workers[model_id].task_q.put(UnloadModelTask())

            # Wait for unload
            self.workers[model_id].process.join()
            logger.info(f"Worker {model_id} stopped...")

            # Remove the worker from the dictionary
            del self.workers[model_id]

    def stop_all(self):
        """
        Send a inference task to the corresponding modelfile worker
        """
        # Loop over all the workers to stop/unload
        for model_id in list(self.workers.keys()):
            self.stop_worker(model_id)

    def clear_cache_worker(self, model_id: str):
        """
        Clear the KV chache of a modelfile worker

        Args:
            model_name (str): Workers to clear cache.

        """
        if model_id in self.workers.keys():
            # Get the queue of tasks of the worker

            # Send the abort task of the modelfile if currently is running some inference
            self.workers[model_id].task_q.put(ClearCacheTask())

    def inference(self, model_id: str, model_input):
        """
        Send a inference task to the corresponding modelfile worker

        Args:
            model_id (str): Model name to invoke
            model_input (str): Input of the modelfile

        """

        # TODO: passer les options avec la task
        if model_id in self.workers.keys():
            # Send the inference task
            self.send_task(
                model_id,
                InferenceTask(
                    RKLLMInferMode.RKLLM_INFER_GENERATE,
                    RKLLMInputType.RKLLM_INPUT_TOKEN,
                    model_input,
                ),
            )

    def embedding(self, model_id: str, model_input):
        """
        Send a prepare embedding task to the corresponding modelfile worker

        Args:
            model_id (str): Model name to invoke
            model_input (str): Input of the modelfile

        """
        if model_id in self.workers.keys():
            # Send the inference task
            self.send_task(
                model_id,
                EmbeddingTask(
                    RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
                    RKLLMInputType.RKLLM_INPUT_TOKEN,
                    model_input,
                ),
            )

    def multimodal(self, model_id: str, prompt_input, images):
        """
        Send a inference task to the corresponding modelfile worker for multimodal input

        Args:
            model_id (str): Model name to invoke
            prompt_input (str): Input of the modelfile
            image_embed (np.ndarray): Image embedding
            n_image_tokens (int): Number of image tokens
            image_width (int): Width of the image
            image_height (int): Height of the image

        """

        # TODO: import from backend
        from .rknn import IMAGE_TOKEN_NUM, IMAGE_WIDTH, IMAGE_HEIGHT

        if model_id in self.workers.keys():
            # Prepare the image input embed for multimodal
            image_embed = self.get_image_embed(model_id, images)

            # Check if the image was encoded correctly
            if image_embed is None:
                # Error encoding the image. Return
                raise RuntimeError(
                    f"Unexpected error encoding image for modelfile : {model_id}"
                )

            # Prepare all the inputs for the multimodal inference
            model_input = (
                prompt_input,
                image_embed,
                IMAGE_TOKEN_NUM,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
            )

            # Send the inference task
            self.send_task(
                model_id,
                InferenceTask(
                    RKLLMInferMode.RKLLM_INFER_GENERATE,
                    RKLLMInputType.RKLLM_INPUT_MULTIMODAL,
                    model_input,
                ),
            )

    def get_image_embed(self, model_id: str, images) -> None:
        """
        Send a vision encoder task to the corresponding modelfile worker

        Args:
            model_id (str): Model name to invoke
            image_path (str): Path of the image to encode

        """
        if model_id in self.workers.keys():
            # Specify the RKNN core to use to encode the image
            core_mask = RKNN_NPU_CORE_ALL  # All cores available

            # Find a temporally base domain id to load the encoder Model
            base_domain_id = self.get_available_base_domain_id(reverse_order=True)

            # Get the path of the vision encoder modelfile
            from core.processing.images.images_utils import get_encoder_model_path

            model_encoder_path = get_encoder_model_path(model_name)

            # Check if the encoder modelfile is available
            if model_encoder_path is None:
                # No vision encoder modelfile available for this RKLLM modelfile
                raise RuntimeError(
                    f"No encoder modelfile (.rknn) found for : {model_name}"
                )

            # Get the image path/base64/url from the request
            image_path = images[
                len(images) - 1
            ]  # For now, only one image supported (the last one)

            # Prepare the input for the vision encoder
            model_input = (model_encoder_path, core_mask, base_domain_id, image_path)

            # Send the Encoder task of the image
            self.send_task(model_id, VisionEncoderTask(model_input))

            # Wait to confirm output of the image encoder
            image_embed = core.config.config_utils.get()

            if isinstance(image_embed, str) and image_embed == Tasks.WORKER_TASK_ERROR:
                # Error ENcoding the image. Return
                return None

            # Return the image encoded
            return image_embed

    def get_finished_inference_token(self):
        """
        Return the finish token for inference task

        Returns:
            str: Token for finished inference.
        """
        return WORKER_TASK_FINISHED


worker_managers: List[WorkerManager] = []
monitor_thread_name = None


def start_models_monitor(interval=60) -> str:
    """
    Start a threat to monitor expired models to unload them from memory

    Args:
        interval: Interval between check
    """

    def execute():
        while True:
            try:
                # Call the process to unload expired models
                for worker_manager in worker_managers:
                    worker_manager.unload_expired_models()
                # Wait for the next execution
                time.sleep(interval)  # Check every 60 seconds expired models
            except Exception as e:
                logger.error(f"Exception in monitor models: {e}")

    # Iniciar el hilo como daemon (no bloquea al final del programa)
    thread = threading.Thread(target=execute, daemon=True)
    thread.start()
    logger.info("Models Monitor running.")
    return thread.name


def get_worker_manager(backend_type: BackendType) -> WorkerManager:
    for worker_manager in worker_managers:
        if worker_manager.backend_type == backend_type:
            return worker_manager

    worker_manager = WorkerManager(backend_type=backend_type)
    worker_managers.append(worker_manager)

    # Start the monitor of running models
    global monitor_thread_name
    if monitor_thread_name is None:
        monitor_thread_name = start_models_monitor()

    return worker_manager
