import datetime
import json
import re
import threading
import time
from typing import Union, Dict, List

import core.config.config_utils
import core.model
from core.api.parameters import Message, Role
from core.backends.backend import Backend
from core.model.ModelConfig import FullModelParameters
from core.model.ModelFile import ModelFile
from core.processing import APIHandler
from core.processing.workers.Worker import Worker
from core.processing.WorkerManager import get_worker_manager
from core.processing.endpoints import logger
from core.processing.endpoints.EndpointHandler import EndpointHandler
from core.processing.format_spec.formatting import create_format_instruction, validate_format_response


class GenerateEndpointHandler(EndpointHandler):
    """Handler for /api/generate endpoint requests"""

    @staticmethod
    def format_streaming_chunk(
            model_id: str,
            token, is_final=False, metrics=None, format_data=None
    ):
        """Format a streaming chunk for generate endpoint"""
        chunk = {
            "model": model_id,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "response": token if not is_final else "",
            "done": is_final,
        }

        if is_final:
            chunk["done_reason"] = "stop"
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
    def format_complete_response(model_name, complete_text, metrics, format_data=None):
        """Format a complete non-streaming response for generate endpoint"""
        response = {
            "model": model_name,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "response": complete_text
            if not (format_data and "cleaned_json" in format_data)
            else format_data["cleaned_json"],
            "done_reason": "stop",
            "done": True,
            "total_duration": metrics["total"],
            "load_duration": metrics["load"],
            "prompt_eval_count": core.config.config_utils.get("prompt_tokens", 0),
            "prompt_eval_duration": metrics["prompt_eval"],
            "eval_count": core.config.config_utils.get("token_count", 0),
            "eval_duration": metrics["eval"],
            "context": [],
        }

        return response

    @classmethod
    def handle_request(
        cls,
        model_worker: Worker,
        api_handler: APIHandler,
        modelfile: ModelFile,
        prompt: str,
        system: str,
        stream: bool,
        options: FullModelParameters,
        enable_thinking: bool = False,
        format_spec=None,
    ):
        """Process a generate request with proper format handling"""
        messages: List[Message] = [Message(role=Role.USER, content=prompt)]

        model_id: str = modelfile.model_id

        if system is None or system == "":
            system = modelfile.SYSTEM

        if cls.DEBUG_MODE:
            logger.debug(
                f"GenerateEndpointHandler: processing request for {model_id}"
            )
            logger.debug(f"Format spec: {format_spec}")

        try:
            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction and messages:
                    if cls.DEBUG_MODE:
                        logger.debug(
                            f"Adding format instruction to prompt: {format_instruction}"
                        )
                    if isinstance(messages[0].content, str):
                        messages[0].content += format_instruction
                    else:
                        messages[0].content.append(format_instruction)

            # Any, Union[list[int], Dict], int
            _, prompt_tokens, prompt_token_count = cls.prepare_prompt(
                modelfile=modelfile,
                messages=messages,
                system=system,
                enable_thinking=enable_thinking
            )

            if stream:
                return cls.handle_streaming(
                    model_worker,
                    api_handler,
                    model_id,
                    prompt_tokens,
                    prompt_token_count,
                    format_spec,
                )
            else:
                return cls.handle_complete(
                    model_worker,
                    api_handler,
                    model_id,
                    prompt_tokens,
                    prompt_token_count,
                    format_spec,
                )

    @classmethod
    def handle_streaming(
            cls,
            model_backend: Backend,
            api_handler: APIHandler,
            model_id: str,
            prompt_tokens, prompt_token_count, format_spec
    ):
        """Handle streaming generate response"""

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

            while not thread_finished or not final_sent:
                tokens_processed = False

                while len(variables.global_text) > 0:
                    tokens_processed = True
                    count += 1
                    token = variables.global_text.pop(0)

                    if count == 1:
                        prompt_eval_time = time.time()

                    complete_text += token

                    if variables.global_status != 1:
                        chunk = cls.format_streaming_chunk(model_name, token)
                        yield f"{json.dumps(chunk)}\n"
                    else:
                        pass

                thread_model.join(timeout=0.005)
                thread_finished = not thread_model.is_alive()

                if thread_finished and not final_sent:
                    final_sent = True

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
                        model_name, "", True, metrics, format_data
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
            format_spec,
            images=None
    ):

        """Handle complete generate response"""

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
        if format_spec and complete_text:
            if DEBUG_MODE:
                logger.debug(f"Validating format for complete text: {complete_text[:300]}...")
                if isinstance(format_spec, str):
                    logger.debug(f"Format is string type: {format_spec}")

            success, parsed_data, error, cleaned_json = validate_format_response(complete_text, format_spec)

            if not success and isinstance(format_spec, str) and format_spec.lower() == 'json':
                if DEBUG_MODE:
                    logger.debug("Simple JSON format validation failed, attempting additional extraction")

                json_pattern = r'\{[\s\S]*?\}'
                matches = re.findall(json_pattern, complete_text)

                for match in matches:
                    try:
                        fixed = match.replace("'", '"')
                        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
                        test_parsed = json.loads(fixed)
                        success, parsed_data, error, cleaned_json = True, test_parsed, None, fixed
                        if DEBUG_MODE:
                            logger.debug(f"Extracted valid JSON using additional methods: {cleaned_json}")
                        break
                    except:
                        continue

            elif not success and isinstance(format_spec, dict) and format_spec.get('type') == 'object':
                if DEBUG_MODE:
                    logger.debug(f"Initial validation failed: {error}. Trying to fix JSON...")

                json_pattern = r'\{[\s\S]*?\}'
                matches = re.findall(json_pattern, complete_text)

                for match in matches:
                    fixed = match.replace("'", '"')
                    fixed = re.sub(r'(\w+):', r'"\1":', fixed)

                    try:
                        test_parsed = json.loads(fixed)
                        required_fields = format_spec.get('required', [])
                        has_required = all(field in test_parsed for field in required_fields)

                        if has_required:
                            success, parsed_data, error, cleaned_json = validate_format_response(fixed, format_spec)
                            if success:
                                if DEBUG_MODE:
                                    logger.debug(f"Fixed JSON validation succeeded: {cleaned_json}")
                                break
                    except:
                        continue

            if DEBUG_MODE:
                logger.debug(f"Format validation result: success={success}, error={error}")
                if cleaned_json and success:
                    logger.debug(f"Cleaned JSON: {cleaned_json}")
                elif not success:
                    logger.debug(f"JSON validation failed, response will not include parsed data")

            if success and parsed_data:
                if isinstance(format_spec, str):
                    format_type = format_spec
                else:
                    format_type = format_spec.get("type", "json") if isinstance(format_spec, dict) else "json"

                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json
                }

        response = cls.format_complete_response(model_name, complete_text, metrics, format_data)

        if DEBUG_MODE and format_data:
            logger.debug(f"Created formatted response with JSON content")

        return jsonify(response), 200






        """Handle complete generate response"""
        start_time = time.time()
        prompt_eval_time = None

        thread_model = threading.Thread(target=model_backend.run, args=(prompt_tokens,))
        thread_model.start()

        count = 0
        complete_text = ""

        while thread_model.is_alive() or len(variables.global_text) > 0:
            while len(variables.global_text) > 0:
                count += 1
                token = variables.global_text.pop(0)

                if count == 1:
                    prompt_eval_time = time.time()

                complete_text += token

            thread_model.join(timeout=0.005)

        metrics = cls.calculate_durations(start_time, prompt_eval_time)
        metrics.prompt_tokens = prompt_token_count
        metrics.token_count = count

        format_data = None
        if format_spec and complete_text:
            if cls.DEBUG_MODE:
                logger.debug(
                    f"Validating format for complete text: {complete_text[:300]}..."
                )
                if isinstance(format_spec, str):
                    logger.debug(f"Format is string type: {format_spec}")

            success, parsed_data, error, cleaned_json = validate_format_response(
                complete_text, format_spec
            )

            if (
                not success
                and isinstance(format_spec, str)
                and format_spec.lower() == "json"
            ):
                if cls.DEBUG_MODE:
                    logger.debug(
                        "Simple JSON format validation failed, attempting additional extraction"
                    )

                json_pattern = r"\{[\s\S]*?\}"
                matches = re.findall(json_pattern, complete_text)

                for match in matches:
                    try:
                        fixed = match.replace("'", '"')
                        fixed = re.sub(r"(\w+):", r'"\1":', fixed)
                        test_parsed = json.loads(fixed)
                        success, parsed_data, error, cleaned_json = (
                            True,
                            test_parsed,
                            None,
                            fixed,
                        )
                        if cls.DEBUG_MODE:
                            logger.debug(
                                f"Extracted valid JSON using additional methods: {cleaned_json}"
                            )
                        break
                    except:
                        continue

            elif (
                not success
                and isinstance(format_spec, dict)
                and format_spec.get("type") == "object"
            ):
                if cls.DEBUG_MODE:
                    logger.debug(
                        f"Initial validation failed: {error}. Trying to fix JSON..."
                    )

                json_pattern = r"\{[\s\S]*?\}"
                matches = re.findall(json_pattern, complete_text)

                for match in matches:
                    fixed = match.replace("'", '"')
                    fixed = re.sub(r"(\w+):", r'"\1":', fixed)

                    try:
                        test_parsed = json.loads(fixed)
                        required_fields = format_spec.get("required", [])
                        has_required = all(
                            field in test_parsed for field in required_fields
                        )

                        if has_required:
                            success, parsed_data, error, cleaned_json = (
                                validate_format_response(fixed, format_spec)
                            )
                            if success:
                                if cls.DEBUG_MODE:
                                    logger.debug(
                                        f"Fixed JSON validation succeeded: {cleaned_json}"
                                    )
                                break
                    except:
                        continue

            if cls.DEBUG_MODE:
                logger.debug(
                    f"Format validation result: success={success}, error={error}"
                )
                if cleaned_json and success:
                    logger.debug(f"Cleaned JSON: {cleaned_json}")
                elif not success:
                    logger.debug(
                        "JSON validation failed, response will not include parsed data"
                    )

            if success and parsed_data:
                if isinstance(format_spec, str):
                    format_type = format_spec
                else:
                    format_type = (
                        format_spec.get("type", "json")
                        if isinstance(format_spec, dict)
                        else "json"
                    )

                format_data = {
                    "format_type": format_type,
                    "parsed": parsed_data,
                    "cleaned_json": cleaned_json,
                }

        response = cls.format_complete_response(
            model_id, complete_text, metrics, format_data
        )

        if cls.DEBUG_MODE and format_data:
            logger.debug("Created formatted response with JSON content")

        retunr api_handler.format_response(
            response=response,
            prompt: str,
            usage_prompt_tokens: int,
            counters: Counters.from_endpoint_metrics(metrics),
            shared_data: SharedData
        )
        return jsonify(response), 200
