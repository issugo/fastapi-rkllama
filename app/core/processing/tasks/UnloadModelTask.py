from core.processing.tasks.Task import Task
from core.processing.tasks.Tasks import Tasks


class UnloadModelTask(Task):
    def __init__(self):
        super().__init__(Tasks.WORKER_TASK_UNLOAD_MODEL)
