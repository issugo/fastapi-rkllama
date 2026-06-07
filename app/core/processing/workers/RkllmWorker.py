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
from core.config import config_utils
from core.config.PathsConfig import PATH_KEY

logger = logging.getLogger("rkllama.worker")


class RkllmWorker(Worker):
    BACKEND_TYPE = BackendType.RKLLM

    # TODO: manage lora_model_path=None
    def create_worker_process(self) -> Process | None:
        """
        Creates the process of the worker
        """

        # Define the process for the worker
        self.process = Process(
            target=RkllmWorker.run,
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
            target=self.model_backend.run,
            args=(model_input,),
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
            import os

            settings = config_utils.get_settings()

            use_simu = os.environ.get("RKLLAMA_SIMULATE", "").lower() in ["1", "true"]

            if not use_simu:
                try:
                    from core.backends.rkllm.rkllm_backend import (
                        callback_type,
                        RKLLMBackend,
                    )
                    from core.backends.rkllm.callback import callback_impl
                except (ImportError, OSError) as e:
                    logger.warning(
                        f"Could not load real RKLLM backend ({e}). Falling back to simulation."
                    )
                    use_simu = True

            if use_simu:
                from core.backends.rkllm_simu_backend.rkllm_simu_backend import (
                    RkllmSimuBackend,
                )

                model_backend = RkllmSimuBackend(
                    model=model,
                    options=options,
                    base_domain_id=base_domain_id,
                )
            else:
                from core.backends.rkllm.rkllm_backend import (
                    callback_type,
                    RKLLMBackend,
                )
                from core.backends.rkllm.callback import callback_impl

                # Connect the callback function between Python and C++ independently for each worker
                callback = callback_type(callback_impl)

                # Resolve prompt cache path from settings instead of parameter
                prompt_cache_path = settings.get_path(PATH_KEY.PROMPT_CACHE)

                model_backend = RKLLMBackend(
                    model=model,
                    options=options,
                    base_domain_id=base_domain_id,
                    prompt_cache_path=prompt_cache_path,
                    # lora_model_path,
                )

            # Announce the creation of the RKLLM modelfile finished
            result_queue.put(Tasks.WORKER_TASK_FINISHED)

        except Exception as e:
            import traceback
            import sys

            sys.stderr.write(
                f"WORKER_CREATION_FAILED_TRACEBACK:\n{traceback.format_exc()}\n"
            )
            sys.stderr.flush()
            logger.error(f"Failed creating the worker for model '{name}': {str(e)}")
            # Announce the creation of the RKLLM modelfile in memory
            result_queue.put(Tasks.WORKER_TASK_ERROR)
            return

        worker.task_wait_loop(model_backend, name, result_queue, task_queue)
