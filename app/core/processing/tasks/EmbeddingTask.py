from typing import Any
from core.processing.tasks.Task import Task
from core.processing.tasks.Tasks import Tasks


class EmbeddingTask(Task):
    def __init__(
        self,
        inference_mode: Any = None,
        model_input_type: Any = None,
        model_input: Any = None,
    ):
        super().__init__(
            Tasks.WORKER_TASK_EMBEDDING, inference_mode, model_input_type, model_input
        )
