from typing import Any
from core.processing.endpoints.EndpointHandler import EndpointHandler
from core.processing.tasks.Task import Task
from core.processing.tasks.Tasks import Tasks

class VisionEncoderTask(Task):
    def __init__(self, model_input: Any = None):
        super().__init__(Tasks.WORKER_TASK_VISION_ENCODER, model_input=model_input)
