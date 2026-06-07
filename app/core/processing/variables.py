import threading
from typing import List, Dict, Any

# NOTE: This module is deprecated. Do not add new variables here.
# It is kept solely for legacy references and backward compatibility.
# each times something is needed from this file, please consider refactoring to use equivalent variable (usualy defined as parameters)
# remove unused variables to avoid confusion

class MockWorkerManager:
    workers: Dict[str, Any] = {}
    def exists_model_loaded(self, model_name: str) -> bool:
        return False
    def get_result(self, model_name: str):
        return None
    def get_finished_inference_token(self):
        return ""
    def inference(self, model_name: str, prompt_tokens: Any):
        pass
    def multimodal(self, model_name: str, prompt_tokens: Any, images: Any):
        pass
    def clear_cache_worker(self, model_name: str):
        pass

worker_manager_rkllm = MockWorkerManager()
verrou = threading.Lock()
global_text: List[str] = []
global_status: int = 0
model_config: Dict[str, Any] = {}
generation_complete: bool = False
debug_mode: bool = False
stream_stats: Dict[str, int] = {
    "total_requests": 0,
    "successful_responses": 0,
    "failed_responses": 0,
    "incomplete_streams": 0,
}
