import json
import time
from threading import Thread
from typing import Any, List

from pydantic import BaseModel

from core.model.Model import ModelSharedData
from core.parameters import Message
from core.parameters.rkllama_responses import RKllamaResponse, RKllamaChoice
from core.processing.formatting import validate_format_response, create_format_instruction
from core.processing.Handler import Counters, DataFormat, Handler, SharedData
from core.processing import logger
from loggers import DEBUG_MODE


class RKllamaHandler(Handler):

    def __init__(self):
        super().__init__("text/plain")

    def new_response(self):
        return RKllamaResponse()

    def generate(self, counters: Counters, shared_data: SharedData, response: RKllamaResponse, model_thread: Thread, model_shared_data: ModelSharedData):
        counters.count = 0
        counters.start = time.time()
        counters.prompt_eval_end_time = None
        counters.final_message_sent = False  # Track if we've sent the final message

        # Initialize accumulated text for JSON format validation
        counters.complete_text = ""
        counters.tokens_since_last_response = (
            0  # Track tokens since last response sent
        )


        model_thread.start()

        thread_model_finished = False

        while not thread_model_finished:
            logger.debug("rkllm thread running")

            while len(model_shared_data.global_text) > 0:
                current_token = model_shared_data.global_text.pop(0)
                response.choices = [
                    RKllamaChoice(
                        role= "assistant",
                        content= current_token,
                        finish_reason= ("stop"
                        if model_shared_data.global_status == 1
                        else None)
                    )
                ]
                response.usage.completion_tokens = counters.count
                response.usage.total_tokens += 1

                # Process format in the final chunk
                if model_shared_data.global_status == 1 and shared_data.data_format.format_spec:
                    shared_data.success, shared_data.parsed_data, shared_data.error, shared_data.cleaned_json = (
                        validate_format_response(counters.complete_text, shared_data.data_format.format_spec)
                    )
                    if shared_data.success and shared_data.parsed_data:
                        response.choices[0].format = shared_data.data_format.format_spec
                        response.choices[0].parsed = shared_data.parsed_data

                # Send the response
                yield f"{response.model_dump_json()}\n\n"

            logger.debug("sleeping")
            time.sleep(0.005)
            logger.debug("joining")
            model_thread.join(timeout=0.005)
            thread_model_finished = not model_thread.is_alive()

        logger.info("rkllm thread finished")

    def format_response(self, response: RKllamaResponse, prompt : str, usage_prompt_tokens: int, counters: Counters, shared_data: SharedData) -> dict[
        str | Any, str | None | dict[str, str | Any] | bool | int | Any]:

        # Define the structure of the returned response
        response.created = counters.created_time
        response.usage.prompt_tokens = usage_prompt_tokens
        response.usage.total_tokens = usage_prompt_tokens

        # Standard RKLLAMA API response
        response.choices[0].content = (shared_data.cleaned_json
                if shared_data.success and shared_data.cleaned_json
                else counters.complete_text)

        # Add format information if available
        if shared_data.success and shared_data.parsed_data:
            response.choices[0].format = shared_data.data_format.format_spec
            response.choices[0].parsed = shared_data.parsed_data

        # Update token counts
        response.usage.completion_tokens = counters.count
        response.usage.total_tokens = (
                response.usage.prompt_tokens + counters.count
        )

        # Calculate tokens per second if we have meaningful duration
        if counters.eval_duration > 0:
            response.usage.tokens_per_second = round(
                counters.count / counters.eval_duration, 2
            )
        return response.__dict__

