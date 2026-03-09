import logging
from datetime import datetime
from multiprocessing import Process, Queue
from abc import ABC, abstractmethod
from typing import Any

from core.backends.backend import BackendType, Backend
from core.backends.rkllm.rkllm_backend import RKLLMBackend
from core.model.Model import Model
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.processing.BaseDomainId import BaseDomainId
from core.processing.tasks.Tasks import Tasks
from core.processing.workers.WorkerModelInfo import WorkerModelInfo

logger = logging.getLogger("rkllama.worker")


# Class to manage the information for running RKLLM models
class Worker(ABC):
    BACKEND_TYPE: BackendType | None = None
    _backend_type: BackendType | None = None

    def __init__(self, modelfile: ModelFile, full_model_parameters: FullModelParameters, base_domain_id: BaseDomainId):
        self.worker_model_info = WorkerModelInfo(modelfile=modelfile, base_domain_id=base_domain_id)
        self.full_model_parameters = full_model_parameters
        self.process = None
        self.task_q = Queue()
        self.result_q = Queue()

    @property
    def modelfile(self) -> ModelFile:
        return self.worker_model_info.modelfile

    @property
    def base_domain_id(self) -> BaseDomainId:
        return self.worker_model_info.base_domain_id

    @property
    def backend_type(self) -> BackendType:
        if self.BACKEND_TYPE is not None:
            return self.BACKEND_TYPE
        if self._backend_type is None:
            self._backend_type = BackendType.from_model_type(self.modelfile.model_type)
        return self._backend_type


    @abstractmethod
    def create_worker_process(self) -> Process | None:
        """
        Creates the process of the worker
        """
        pass


    @abstractmethod
    def create_inference_thread(self, inference_mode, model_input, model_input_type) -> Any:
        pass


    def task_wait_loop(self, model_backend: Backend, name: str, result_queue: Queue, task_queue: Queue):
        # Loop to wait for tasks
        while True:

            try:

                # Get the instruction to the worker
                task_obj = task_queue.get()
                task = task_obj.task
                inference_mode = task_obj.inference_mode
                model_input_type = task_obj.model_input_type
                model_input = task_obj.model_input

                if task == Tasks.WORKER_TASK_UNLOAD_MODEL:
                    logger.info(f"Unloading model {name}...")
                    # Unload the modelfile
                    model_backend.release()

                    # Exit the loop of the worker to finish the process
                    break

                elif task == Tasks.WORKER_TASK_CLEAR_CACHE:
                    logger.info(f"Clearing KV cache for model {name}...")
                    # CLear the cache of the modelfile
                    model_backend.clear_cache()

                elif task == Tasks.WORKER_TASK_ABORT_INFERENCE:
                    logger.info(f"Aborting inference for model {name}...")
                    # Abort the inference of the modelfile
                    model_backend.abort()

                elif task == Tasks.WORKER_TASK_INFERENCE:
                    logger.info(f"Running inference for model {name}...")
                    # Run inference

                    # TODO: use workermanager
                    thread_model = self.create_inference_thread(inference_mode, model_input, model_input_type)
                    thread_model.start()

                    # Looping until execution of the thread
                    thread_finished = False
                    while not thread_finished:
                        tokens_processed = False
                        while len(global_text) > 0:
                            tokens_processed = False
                            token = global_text.pop(0)
                            result_queue.put(token)

                        # Update status of the thread
                        thread_model.join(timeout=0.005)
                        thread_finished = not thread_model.is_alive()

                        # If inference not started yet, wait some time to start.
                        if not tokens_processed:
                            time.sleep(0.01)

                    # Send final signal of the inference
                    result_queue.put(Tasks.WORKER_TASK_FINISHED)

                # elif task == Tasks.WORKER_TASK_EMBEDDING:
                #     logger.info(f"Running embedding for model {name}...")
                #     # Run inference
                #     thread_model = threading.Thread(target=model_rkllm.run,
                #                                     args=(inference_mode, model_input_type, model_input,))
                #     thread_model.start()
                #
                #     # Looping until execution of the thread finished
                #     thread_finished = False
                #     while not thread_finished:
                #         # Update status of the thread
                #         thread_model.join(timeout=0.005)
                #         thread_finished = not thread_model.is_alive()
                #
                #     if last_embeddings:
                #         # Send the embedding shapes of the input
                #         result_queue.put(last_embeddings[0])
                #
                # elif task == Tasks.WORKER_TASK_VISION_ENCODER:
                #     logger.info(f"Running vision encoder for model {name}...")
                #     # Run the vision encoder to get the image embedding
                #     rknn_queue = Queue()
                #
                #     # Define the process for the encoder
                #     rknn_process = Process(target=run_encoder, args=(model_input, rknn_queue,))
                #
                #     # Start the encoder worker
                #     rknn_process.start()
                #
                #     # Get the encoded image from the queue
                #     img_encoded = rknn_queue.get()
                #
                #     # Terminate the process encoder after use
                #     rknn_process.terminate()
                #
                #     # Send the encoded image
                #     result_queue.put(img_encoded)
                else:
                    result_queue.put(f"Unknown task: {task}")
                    # Send final signal of the inference
                    result_queue.put(Tasks.WORKER_TASK_FINISHED)
            except Exception as e:
                logger.error(f"Failed executing task the worker for model '{name}' for task '{task}': {str(e)}")
                # Announce the creation of the RKLLM modelfile in memory
                result_queue.put(Tasks.WORKER_TASK_ERROR)


