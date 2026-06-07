import logging
import threading
from datetime import datetime
from multiprocessing import Process, Queue
from typing import Any

from core.backends.backend import BackendType
from core.processing.workers.Worker import Worker
from core.model.Model import Model
from core.model.ModelConfig import FullModelParameters
from core.processing.BaseDomainId import BaseDomainId
from core.processing.tasks.Tasks import Tasks

logger = logging.getLogger("rkllama.worker")


class RknnWorker(Worker):
    BACKEND_TYPE = BackendType.RKNN

    # TODO: manage lora_model_path=None
    def create_worker_process(self) -> Process | None:
        """
        Creates the process of the worker
        """

        # Define the process for the worker
        self.process = Process(
            target=Worker.run,
            args=(
                f"{self.modelfile.model_id}_{str(datetime.now())}",
                self,
                self.task_q,
                self.result_q,
                self.modelfile.model,
                self.full_model_parameters,
                self.base_domain_id,
            ),
        )

        # Start the worker
        self.process.start()

        # Wait to confirm initialization
        creation_status = self.result_q.get()

        if creation_status == Tasks.WORKER_TASK_ERROR:
            # Error loading the RKLLM Model. Wait for the worker to exit
            self.process.terminate()
            return None

        # Success loading the modelfile
        return self.process

    def create_inference_thread(
        self, inference_mode, model_input, model_input_type
    ) -> Any:
        return threading.Thread(
            target=model_rkllm.run,
            args=(
                inference_mode,
                model_input_type,
                model_input,
            ),
        )

    @staticmethod
    def run(
        name: str,
        worker: Worker,
        task_queue: Queue,
        result_queue: Queue,
        model: Model,
        options: FullModelParameters,
        base_domain_id: BaseDomainId,
    ):
        # Define the modelfile used by the worker
        try:
            # model_backend = RKNNBackend(callback, model_path, model_dir, options, lora_model_path, prompt_cache_path,
            #                base_domain_id)
            pass

            # Announce the creation of the RKLLM modelfile failed
            result_queue.put(Tasks.WORKER_TASK_FINISHED)

        except Exception as e:
            logger.error(f"Failed creating the worker for model '{name}': {str(e)}")
            # Announce the creation of the RKLLM modelfile in memory
            result_queue.put(Tasks.WORKER_TASK_ERROR)
            return

        worker.task_wait_loop(model_backend, name, result_queue, task_queue)
