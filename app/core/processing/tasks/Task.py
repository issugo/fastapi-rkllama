from typing import Any
from abc import ABC
from core.processing.tasks.Tasks import Tasks


class Task(ABC):
    """Base class for tasks sent to the worker"""

    def __init__(
        self,
        task: Tasks,
        inference_mode: Any = None,
        model_input_type: Any = None,
        model_input: Any = None,
    ):
        self.task = task
        self.inference_mode = inference_mode
        self.model_input_type = model_input_type
        self.model_input = model_input
