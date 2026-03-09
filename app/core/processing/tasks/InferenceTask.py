from typing import Any
from core.processing.endpoints.EndpointHandler import EndpointHandler
from core.processing.tasks.Task import Task
from core.processing.tasks.Tasks import Tasks

class InferenceTask(Task):
    def __init__(self, inference_mode: Any = None, model_input_type: Any = None, model_input: Any = None):
        super().__init__(Tasks.WORKER_TASK_INFERENCE, inference_mode, model_input_type, model_input)

