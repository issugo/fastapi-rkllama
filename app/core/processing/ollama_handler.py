
class OllamaHandler(Handler):

    def __init__(self):
        super().__init__("application/x-ndjson")

    def format_response(self, cleaned_json, complete_text: str, prompt : str, created_time: int, count: int, eval_duration: float,
                              usage_prompt_tokens: int, parsed_data, format_spec, load_duration: float,
                              prompt_eval_duration: float, success, total_duration: float) -> dict[
        str | Any, str | None | dict[str, str | Any] | bool | int | Any]:
        return {
            "model": GLOBAL_STATE.loaded_model_hfpath,
            "created_at": datetime.datetime(created_time).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "message": {
                "role": "assistant",
                # Use only the clean JSON text if available, otherwise use complete response
                "content": cleaned_json
                if success and cleaned_json
                else complete_text,
            },
            "done_reason": "stop",  # Always add done_reason for completed responses
            "done": True,
            # Add all required duration fields in nanoseconds
            "total_duration": int(total_duration * 1_000_000_000),
            "load_duration": int(
                load_duration * 1_000_000_000
            ),  # Fixed 100ms
            "prompt_eval_count": usage_prompt_tokens,
            "prompt_eval_duration": int(
                prompt_eval_duration * 1_000_000_000
            ),
            "eval_count": count,
            "eval_duration": int(eval_duration * 1_000_000_000),
        }

