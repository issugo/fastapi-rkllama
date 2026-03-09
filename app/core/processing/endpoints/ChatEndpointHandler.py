import datetime
import json
import threading
import time
from typing import List, Union, Dict

import core.config.config_utils
import core.model
from core.api.parameters import Message, Role
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.processing import APIHandler
from core.processing.workers.Worker import Worker
from core.processing.WorkerManager import get_worker_manager, WorkerManager
from core.processing.endpoints.EndpointHandler import EndpointHandler
from core.processing.format_spec.formatting import create_format_instruction, validate_format_response
from core.processing.tools.tools_utils import get_tool_calls
from src import variables as variables
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
                        "total_duration": metrics["total"],
                        "load_duration": metrics["load"],
                        "prompt_eval_count": core.config.config_utils.get("prompt_tokens", 0),
                        "prompt_eval_duration": metrics["prompt_eval"],
                        "eval_count": core.config.config_utils.get("token_count", 0),
                        "eval_duration": metrics["eval"],
                    }
                )

        return chunk

    @staticmethod
    def format_complete_response(model_name, complete_text, metrics : dict, format_data=None):
        """Format a complete non-streaming response for chat endpoint"""
        response = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "message": {
                "role": "assistant",
                "content": complete_text
                if not (format_data and "cleaned_json" in format_data)
                else format_data["cleaned_json"],
            },
            "done_reason": "stop"
            if not (format_data and "tool_call" in format_data)
            else "tool_calls",
            "done": True,
            "total_duration": metrics["total"],
            "load_duration": metrics["load"],
            "prompt_eval_count": core.config.config_utils.get("prompt_tokens", 0),
            "prompt_eval_duration": metrics["prompt_eval"],
            "eval_count": core.config.config_utils.get("token_count", 0),
            "eval_duration": metrics["eval"],
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
            logger.debug(
                f"ChatEndpointHandler: processing request for {model_id}"
            )
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
                    enable_thinking=enable_thinking
                )

            else:
                if cls.DEBUG_MODE:
                    logger.debug(f"Multimodal request detected. Skipping tokenization.")

                for message in messages:
                    if "images" in message:
                        message.pop(
                            "images")  # Remove images from messages to avoid context length reach with base64 images
                prompt_tokens = f"<image>{str(messages)}"
                prompt_token_count = 0

            # Ollama request handling
            if stream:
                ollama_chunk = cls.handle_streaming(modelfile, prompt_tokens,
                                                    prompt_token_count, format_spec, tools, enable_thinking, images)
                # Use unified handler
                result = api_handler.format_stream_chunk(ollama_chunk)

                # Return Ollama streaming response
                return api_handler.format_stream_response(stream_with_context(result), mimetype="text/event-stream")
            else:
                ollama_response, code = cls.handle_complete(model_name, prompt_tokens,
                                                            prompt_token_count, format_spec, tools, enable_thinking,
                                                            images)

                # Convert Ollama response to OpenAI format
                ollama_response = api_handler.handle_response(ollama_response, stream=stream, is_chat=True)

                # Return Ollama response
                return ollama_response, code


    @classmethod
    def handle_streaming(
        cls,
        modele_rkllm,
        model_name,
        prompt_tokens,
        prompt_token_count,
        format_spec,
        tools,
        enable_thinking,
    ):
        """Handle streaming chat response"""

        def generate():
            thread_model = threading.Thread(
                target=modele_rkllm.run, args=(prompt_tokens,)
            )
            thread_model.start()

            count = 0
            start_time = time.time()
            prompt_eval_time = None
            complete_text = ""
            final_sent = False

            thread_finished = False

            # Tool calls detection
            max_token_to_wait_for_tool_call = (
                100 if tools else 1
            )  # Max tokens to wait for tool call definition
            tool_calls = False
            first_tokens = []
            thinking = enable_thinking
            final_response_tokens = []

            while not thread_finished or not final_sent:
                tokens_processed = False

                while len(GLOBAL_STATE.global_text) > 0:
                    tokens_processed = True
                    count += 1
                    token = variables.global_text.pop(0)

                    if count == 1:
                        prompt_eval_time = time.time()

                        if thinking and "<think>" not in token.lower():
                            token = (
                                "<think>" + token
                            )  # Ensure correct initial format token <think>
                    else:
                        if thinking and "</think>" in token.lower():
                            thinking = False

                    complete_text += token
                    first_tokens.append(token)

                    if not thinking and token != "</think>":
                        final_response_tokens.append(token)

                    if not tool_calls:
                        if len(final_response_tokens) > max_token_to_wait_for_tool_call:
                            if variables.global_status != 1:
                                chunk = cls.format_streaming_chunk(
                                    model_name=model_name, token=token
                                )
                                yield f"{json.dumps(chunk)}\n"
                            else:
                                pass
                        elif (
                            len(final_response_tokens)
                            == max_token_to_wait_for_tool_call
                        ):
                            if variables.global_status != 1:
                                for temp_token in first_tokens:
                                    time.sleep(
                                        0.1
                                    )  # Simulate delay to stream previos tokens
                                    chunk = cls.format_streaming_chunk(
                                        model_name=model_name, token=temp_token
                                    )
                                    yield f"{json.dumps(chunk)}\n"
                            else:
                                pass
                        elif (
                            len(final_response_tokens) < max_token_to_wait_for_tool_call
                        ):
                            if variables.global_status != 1:
                                # Check if tool call founded in th first tokens in the response
                                tool_calls = "<tool_call>" in token
                            else:
                                pass

                thread_model.join(timeout=0.005)
                thread_finished = not thread_model.is_alive()

                if thread_finished and not final_sent:
                    final_sent = True

                    if tool_calls:
                        chunk_tool_call = cls.format_streaming_chunk(
                            model_name=model_name,
                            token=get_tool_calls(complete_text),
                            tool_calls=tool_calls,
                        )
                        yield f"{json.dumps(chunk_tool_call)}\n"
                    elif count < max_token_to_wait_for_tool_call:
                        for temp_token in first_tokens:
                            time.sleep(0.1)  # Simulate delay to stream previos tokens
                            chunk = cls.format_streaming_chunk(
                                model_name=model_name,
                                token=temp_token,
                                tool_calls=tool_calls,
                            )
                            yield f"{json.dumps(chunk)}\n"

                    metrics = cls.calculate_durations(start_time, prompt_eval_time)
                    metrics["prompt_tokens"] = prompt_token_count
                    metrics["token_count"] = count

                    format_data = None
                    if format_spec and complete_text:
                        success, parsed_data, error, cleaned_json = (
                            validate_format_response(complete_text, format_spec)
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
                        model_name=model_name,
                        token="",
                        is_final=True,
                        metrics=metrics,
                        format_data=format_data,
                        tool_calls=tool_calls,
                    )
                    yield f"{json.dumps(final_chunk)}\n"

                if not tokens_processed:
                    time.sleep(0.01)

        return Response(generate(), content_type="application/x-ndjson")

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
            tools = None,
            images = None
    ):
        """Handle complete non-streaming chat response"""
        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False

        count = 0
        complete_text = ""

        worker_manager: WorkerManager = get_worker_manager(model_worker.backend_type)

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            worker_manager.inference(
                model_id=model_id,
                prompt_tokens=prompt_tokens)
        else:
            # Send the task of multimodal inference to the model
            worker_manager.multimodal(
                model_id=model_id,
                prompt_tokens=prompt_tokens,
                images=images)
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
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = count

        format_data = None
        tool_calls = get_tool_calls(complete_text) if tools else None
        if format_spec and complete_text and not tool_calls:
            success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
            if success and parsed_data:
                format_type = (
                    format_spec.get("type", "") if isinstance(format_spec, dict)
                    else "json"
                )
                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json
                }

        if tool_calls:
            format_data = {
                "format_type": "json",
                "parsed": "",
                "cleaned_json": "",
                "tool_call": tool_calls
            }

        response = cls.format_complete_response(model_name, complete_text, metrics, format_data)
        return jsonify(response), 200

    @classmethod
    def handle_complete(cls, model_name, prompt_tokens, prompt_token_count, format_spec, tools, enable_thinking,
                        images=None):
        """Handle complete non-streaming chat response"""

        start_time = time.time()
        prompt_eval_time = None
        thread_finished = False

        count = 0
        complete_text = ""

        # Check if multimodal or text only
        if not images:
            # Send the task of inference to the model
            variables.worker_manager_rkllm.inference(model_name, prompt_tokens)
        else:
            # Send the task of multimodal inference to the model
            variables.worker_manager_rkllm.multimodal(model_name, prompt_tokens, images)
            # Clear the cache to prevent image embedding problems
            variables.worker_manager_rkllm.clear_cache_worker(model_name)

        # Wait for result queue
        result_q = variables.worker_manager_rkllm.get_result(model_name)
        finished_inference_token = variables.worker_manager_rkllm.get_finished_inference_token()

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
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = count

        format_data = None
        tool_calls = get_tool_calls(complete_text) if tools else None
        if format_spec and complete_text and not tool_calls:
            success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)
            if success and parsed_data:
                format_type = (
                    format_spec.get("type", "") if isinstance(format_spec, dict)
                    else "json"
                )
                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json
                }

        if tool_calls:
            format_data = {
                "format_type": "json",
                "parsed": "",
                "cleaned_json": "",
                "tool_call": tool_calls
            }

        response = cls.format_complete_response(model_name, complete_text, metrics, format_data)
        return jsonify(response), 200

