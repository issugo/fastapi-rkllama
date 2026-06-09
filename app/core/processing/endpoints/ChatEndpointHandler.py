import datetime
import json
import time
from typing import List, Union, Dict

from starlette.responses import JSONResponse, StreamingResponse

import core.config.config_utils
import core.model
from core.api.parameters import Message, Role
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.processing import APIHandler
from core.processing.workers.Worker import Worker
from core.processing.endpoints.EndpointHandler import EndpointHandler
from core.processing.format_spec.formatting import (
    create_format_instruction,
    validate_format_response,
)
from core.processing.tools.tools_utils import get_tool_calls
from core.processing.endpoints import logger


class ChatEndpointHandler(EndpointHandler):
    """Handler for /api/chat endpoint requests"""

    @staticmethod
    def format_streaming_chunk(
        model_name,
        token,
        is_final=False,
        metrics=None,
        format_data=None,
        tool_calls=None,
    ):
        """Format a streaming chunk for chat endpoint"""
        chunk = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {"role": "assistant", "content": token if not is_final else ""},
            "done": is_final,
        }

        if tool_calls:
            chunk["message"]["content"] = ""
            if not is_final:
                chunk["message"]["tool_calls"] = token

        if is_final:
            chunk["done_reason"] = "stop" if not tool_calls else "tool_calls"
            if metrics:
                chunk.update(
                    {
                        "total_duration": metrics.total,
                        "load_duration": metrics.load,
                        "prompt_eval_count": core.config.config_utils.get(
                            "prompt_tokens", 0
                        ),
                        "prompt_eval_duration": metrics.prompt_eval,
                        "eval_count": core.config.config_utils.get("token_count", 0),
                        "eval_duration": metrics.eval,
                    }
                )

        return chunk

    @staticmethod
    def format_complete_response(
        model_name, complete_text, metrics: dict, format_data=None
    ):
        """Format a complete non-streaming response for chat endpoint"""
        response = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": (
                    complete_text
                    if not (format_data and "cleaned_json" in format_data)
                    else format_data["cleaned_json"]
                ),
            },
            "done_reason": (
                "stop"
                if not (format_data and "tool_call" in format_data)
                else "tool_calls"
            ),
            "done": True,
            "total_duration": metrics.total,
            "load_duration": metrics.load,
            "prompt_eval_count": metrics.prompt_tokens,
            "prompt_eval_duration": metrics.prompt_eval,
            "eval_count": metrics.token_count,
            "eval_duration": metrics.eval,
        }

        if format_data and "tool_call" in format_data:
            response["message"]["tool_calls"] = format_data["tool_call"]

        return response

    @classmethod
    def handle_request(
        cls,
        model_worker: Worker,
        api_handler: APIHandler,
        modelfile: ModelFile,
        messages: List[Message],
        system: str,
        stream: bool,
        options: FullModelParameters,
        enable_thinking: bool = False,
        tools=None,
        images=None,
        format_spec=None,
    ):
        """Process a chat request with proper format handling"""

        model_id: str = modelfile.model_id

        if system is None or system == "":
            system = modelfile.SYSTEM

        if cls.DEBUG_MODE:
            logger.debug(f"ChatEndpointHandler: processing request for {model_id}")
            logger.debug(f"Format spec: {format_spec}")

        try:
            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction:
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].role == Role.USER:
                            if isinstance(messages[i].content, str):
                                messages[i].content += format_instruction
                            else:
                                messages[i].content.append(format_instruction)
                            break

            # If Multimodal request, do not use tokenizer
            prompt_tokens = None
            prompt_token_count = None
            if not images:
                # Create the prompts tokens for text only requests
                _, prompt_tokens, prompt_token_count = cls.prepare_prompt(
                    modelfile=modelfile,
                    messages=messages,
                    system=system,
                    tools=tools,
                    enable_thinking=enable_thinking,
                )

            else:
                if cls.DEBUG_MODE:
                    logger.debug("Multimodal request detected. Skipping tokenization.")

                for message in messages:
                    if "images" in message:
                        message.pop(
                            "images"
                        )  # Remove images from messages to avoid context length reach with base64 images
                prompt_tokens = f"<image>{str(messages)}"
                prompt_token_count = 0

            # Ollama request handling
            if stream:
                return cls.handle_streaming(
                    model_worker=model_worker,
                    api_handler=api_handler,
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    prompt_token_count=prompt_token_count,
                    enable_thinking=enable_thinking,
                    format_spec=format_spec,
                    tools=tools,
                    images=images,
                )
            else:
                return cls.handle_complete(
                    model_worker=model_worker,
                    api_handler=api_handler,
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    prompt_token_count=prompt_token_count,
                    enable_thinking=enable_thinking,
                    format_spec=format_spec,
                    tools=tools,
                    images=images,
                )
        except Exception as e:
            logger.error(f"Error in handle_request: {e}", exc_info=True)
            raise e

    @classmethod
    def handle_streaming(
        cls,
        model_worker: Worker,
        api_handler: APIHandler,
        model_id: str,
        prompt_tokens: Union[list[int], Dict],
        prompt_token_count: int,
        enable_thinking: bool,
        format_spec,
        tools=None,
        images=None,
    ):
        """Handle streaming chat response"""

        def generate():
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            count = 0
            thread_finished = False

            from core.processing.WorkerManager import get_worker_manager, WorkerManager

            worker_manager: WorkerManager = get_worker_manager(
                model_worker.backend_type
            )

            # Check if multimodal or text only
            if not images:
                worker_manager.inference(model_id=model_id, model_input=prompt_tokens)
            else:
                worker_manager.multimodal(
                    model_id=model_id, model_input=prompt_tokens, images=images
                )
                worker_manager.clear_cache_worker(model_id=model_id)

            result_q = worker_manager.get_result(model_id=model_id)
            finished_inference_token = worker_manager.get_finished_inference_token()

            # Tool calls detection
            max_token_to_wait_for_tool_call = 100 if tools else 1
            tool_calls = False
            first_tokens = []
            thinking = enable_thinking
            final_response_tokens = []

            while not thread_finished:
                token = result_q.get(timeout=300)
                if token == finished_inference_token:
                    thread_finished = True
                    continue

                count += 1
                if count == 1:
                    prompt_eval_time = time.time()
                    if thinking and "<think>" not in token.lower():
                        token = "<think>" + token
                else:
                    if thinking and "</think>" in token.lower():
                        thinking = False

                complete_text += token
                first_tokens.append(token)

                if not thinking and token != "</think>":
                    final_response_tokens.append(token)

                if not tool_calls:
                    if len(final_response_tokens) > max_token_to_wait_for_tool_call:
                        chunk = cls.format_streaming_chunk(
                            model_name=model_id, token=token
                        )
                        yield (json.dumps(chunk) + "\n").encode("utf-8")
                    elif len(final_response_tokens) == max_token_to_wait_for_tool_call:
                        for temp_token in first_tokens:
                            chunk = cls.format_streaming_chunk(
                                model_name=model_id, token=temp_token
                            )
                            yield (json.dumps(chunk) + "\n").encode("utf-8")
                    elif len(final_response_tokens) < max_token_to_wait_for_tool_call:
                        tool_calls = "<tool_call>" in token

            # Send final response/metrics
            if tool_calls:
                chunk_tool_call = cls.format_streaming_chunk(
                    model_name=model_id,
                    token=get_tool_calls(complete_text),
                    tool_calls=tool_calls,
                )
                yield (json.dumps(chunk_tool_call) + "\n").encode("utf-8")
            elif count < max_token_to_wait_for_tool_call:
                for temp_token in first_tokens:
                    chunk = cls.format_streaming_chunk(
                        model_name=model_id,
                        token=temp_token,
                        tool_calls=tool_calls,
                    )
                    yield (json.dumps(chunk) + "\n").encode("utf-8")

            metrics = cls.calculate_durations(start_time, prompt_eval_time)
            metrics.prompt_tokens = prompt_token_count
            metrics.token_count = count

            format_data = None
            if format_spec and complete_text:
                success, parsed_data, error, cleaned_json = validate_format_response(
                    complete_text, format_spec
                )
                if success and parsed_data:
                    format_type = (
                        format_spec.get("type", "")
                        if isinstance(format_spec, dict)
                        else "json"
                    )
                    format_data = {
                        "format_type": format_type,
                        "parsed": parsed_data,
                        "cleaned_json": cleaned_json,
                    }

            final_chunk = cls.format_streaming_chunk(
                model_name=model_id,
                token="",
                is_final=True,
                metrics=metrics,
                format_data=format_data,
                tool_calls=tool_calls,
            )
            yield (json.dumps(final_chunk) + "\n").encode("utf-8")

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @classmethod
    def handle_complete(
        cls,
        model_worker: Worker,
        api_handler: APIHandler,
        model_id: str,
        prompt_tokens: Union[list[int], Dict],
        prompt_token_count: int,
        enable_thinking: bool,
        format_spec,
        tools=None,
        images=None,
    ):
        """Handle complete non-streaming chat response"""
        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False

        count = 0
        complete_text = ""

        from core.processing.WorkerManager import get_worker_manager, WorkerManager

        worker_manager: WorkerManager = get_worker_manager(model_worker.backend_type)

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            worker_manager.inference(model_id=model_id, model_input=prompt_tokens)
        else:
            # Send the task of multimodal inference to the model
            worker_manager.multimodal(
                model_id=model_id, model_input=prompt_tokens, images=images
            )
            # Clear the cache to prevent image embedding problems
            worker_manager.clear_cache_worker(model_id=model_id)

        # Wait for result queue
        result_q = worker_manager.get_result(model_id=model_id)
        finished_inference_token = worker_manager.get_finished_inference_token()

        while not thread_finished:
            token = result_q.get(timeout=300)  # Block until receive any token
            if token == finished_inference_token:
                thread_finished = True
                continue

            count += 1
            if count == 1:
                prompt_eval_time = time.time()

                if enable_thinking and "<think>" not in token.lower():
                    token = "<think>" + token  # Ensure correct initial format

            complete_text += token

        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics.prompt_tokens = prompt_token_count
        metrics.token_count = count

        format_data = None
        tool_calls = get_tool_calls(complete_text) if tools else None
        if format_spec and complete_text and not tool_calls:
            success, parsed_data, error, cleaned_json = validate_format_response(
                complete_text, format_spec
            )
            if success and parsed_data:
                format_type = (
                    format_spec.get("type", "")
                    if isinstance(format_spec, dict)
                    else "json"
                )
                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json,
                }

        if tool_calls:
            format_data = {
                "format_type": "json",
                "parsed": "",
                "cleaned_json": "",
                "tool_call": tool_calls,
            }

        response = cls.format_complete_response(
            model_id, complete_text, metrics, format_data
        )
        return JSONResponse(content=response, status_code=200)
