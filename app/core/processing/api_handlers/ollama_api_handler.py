import datetime
from importlib.metadata import pass_none
from typing import Any

from core.backends.backend import Backend
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.processing.APIHandler import APIHandler, Counters, SharedData
from core.processing.workers.Worker import Worker
from core.processing.endpoints.ChatEndpointHandler import ChatEndpointHandler
from core.processing.endpoints.GenerateEndpointHandler import GenerateEndpointHandler


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

class OllamaGenerateAPIHandler(OllamaAPIHandler):
    pass

class OllamaChatAPIHandler(OllamaAPIHandler):
    pass

def process_ollama_chat_request(
    model_backend: Backend,
        api_handler: APIHandler,
    modelfile: ModelFile,
    messages,
    system="",
    stream=True,
    format_spec=None,
    options=None,
):
    """Process /api/chat request with correct format"""
    return ChatEndpointHandler.handle_request(
        model_backend=model_backend,
        api_handler=api_handler,
        modelfile=modelfile,
        messages=messages,
        system=system,
        stream=stream,
        format_spec=format_spec,
        options=options,
    )


def process_ollama_generate_request(
    model_worker: Worker,
    api_handler: APIHandler,
    modelfile: ModelFile,
    prompt: str,
    system: str,
    stream: bool,
    options: FullModelParameters,
    enable_thinking: bool,
    images: None | dict[str, str | None] = None,
    format_spec = None,
):
    """Process /api/generate request with correct format"""
    return GenerateEndpointHandler.handle_request(
        model_worker=model_worker,
        api_handler=api_handler,
        modelfile=modelfile,
        prompt=prompt,
        system=system,
        stream=stream,
        options=options,
        format_spec=format_spec,
    )
