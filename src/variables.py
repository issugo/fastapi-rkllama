import threading

from core.config import is_debug_mode

# isLocked now in process as is_locked

# verrou = threading.Lock() # now in Model.usage_lock

model_config = {}  # For storing model-specific configuration
generation_complete = False  # Flag to track completion status
debug_mode = is_debug_mode()
stream_stats = {
    "total_requests": 0,
    "successful_responses": 0,
    "failed_responses": 0,
    "incomplete_streams": 0,  # Streams that didn't receive done=true
}
