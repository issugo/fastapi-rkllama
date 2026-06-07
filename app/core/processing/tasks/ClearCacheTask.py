from core.processing.tasks.Task import Task
from core.processing.tasks.Tasks import Tasks


class ClearCacheTask(Task):
    def __init__(self):
        super().__init__(Tasks.WORKER_TASK_CLEAR_CACHE)
