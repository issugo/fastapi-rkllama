from typing import Any
from core.processing.endpoints.EndpointHandler import EndpointHandler
from core.processing.tasks.Task import Task
from core.processing.tasks.Tasks import Tasks

class AbortInferenceTask(Task):
    def __init__(self):
        super().__init__(Tasks.WORKER_TASK_ABORT_INFERENCE)
