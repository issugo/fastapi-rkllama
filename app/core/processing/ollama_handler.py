import datetime
from typing import Any

from core.processing.APIHandler import APIHandler, Counters, SharedData


class OllamaAPIHandler(APIHandler):

    def __init__(self):
        super().__init__("application/x-ndjson")

    def new_response(self):
        pass



    def format_response(self, response, prompt: str, usage_prompt_tokens: int, counters: Counters, shared_data: SharedData) -> dict[
        str | Any, str | None | dict[str, str | Any] | bool | int | Any]:
        return {
            "model": GLOBAL_STATE.loaded_model_hfpath,
            "created_at": datetime.datetime(counters.created_time).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "message": {
                "role": "assistant",
                # Use only the clean JSON text if available, otherwise use complete response
                "content": shared_data.cleaned_json
                if shared_data.success and shared_data.cleaned_json
                else counters.complete_text,
            },
            "done_reason": "stop",  # Always add done_reason for completed responses
            "done": True,
            # Add all required duration fields in nanoseconds
            "total_duration": int(counters.total_duration * 1_000_000_000),
            "load_duration": int(
                counters.load_duration * 1_000_000_000
            ),  # Fixed 100ms
            "prompt_eval_count": usage_prompt_tokens,
            "prompt_eval_duration": int(
                counters.prompt_eval_duration * 1_000_000_000
            ),
            "eval_count": counters.count,
            "eval_duration": int(counters.eval_duration * 1_000_000_000),
        }

