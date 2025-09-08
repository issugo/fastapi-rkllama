import datetime
import json
import re
import threading
import time

import core.model
from core import config
from core.processing.format_spec.formatting import create_format_instruction, validate_format_response
from src import variables as variables
from src.server_utils import DEBUG_MODE, logger


class GenerateEndpointHandler(EndpointHandler):
    """Handler for /api/generate endpoint requests"""

    @staticmethod
    def format_streaming_chunk(
        model_name, token, is_final=False, metrics=None, format_data=None
    ):
        """Format a streaming chunk for generate endpoint"""
        chunk = {
            "model": model_name,
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
                        "prompt_eval_count": config.get("prompt_tokens", 0),
                        "prompt_eval_duration": metrics["prompt_eval"],
                        "eval_count": config.get("token_count", 0),
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
            "prompt_eval_count": config.get("prompt_tokens", 0),
            "prompt_eval_duration": metrics["prompt_eval"],
            "eval_count": config.get("token_count", 0),
            "eval_duration": metrics["eval"],
            "context": [],
        }

        return response

    @classmethod
    def handle_request(
        cls,
        modele_rkllm,
        model_name,
        prompt,
        system="",
        stream=True,
        format_spec=None,
        options=None,
        enable_thinking=False,
    ):
        """Process a generate request with proper format handling"""
        messages = [{"role": "user", "content": prompt}]

        original_system = core.model.ModelFile.system
        if system:
            core.model.ModelFile.system = system

        if DEBUG_MODE:
            logger.debug(
                f"GenerateEndpointHandler: processing request for {model_name}"
            )
            logger.debug(f"Format spec: {format_spec}")

        try:
            GLOBAL_STATE.global_status = -1

            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction and messages:
                    if DEBUG_MODE:
                        logger.debug(
                            f"Adding format instruction to prompt: {format_instruction}"
                        )
                    messages[0]["content"] += format_instruction

            tokenizer, prompt_tokens, prompt_token_count = cls.prepare_prompt(
                messages=messages, system=system, enable_thinking=enable_thinking
            )

            if stream:
                return cls.handle_streaming(
                    modele_rkllm,
                    model_name,
                    prompt_tokens,
                    prompt_token_count,
                    format_spec,
                )
            else:
                return cls.handle_complete(
                    modele_rkllm,
                    model_name,
                    prompt_tokens,
                    prompt_token_count,
                    format_spec,
                )
        finally:
            core.model.ModelFile.system = original_system

    @classmethod
    def handle_streaming(
        cls, modele_rkllm, model_name, prompt_tokens, prompt_token_count, format_spec
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
        cls, modele_rkllm, model_name, prompt_tokens, prompt_token_count, format_spec
    ):
        """Handle complete generate response"""
        start_time = time.time()
        prompt_eval_time = None

        thread_model = threading.Thread(target=modele_rkllm.run, args=(prompt_tokens,))
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
        metrics["prompt_tokens"] = prompt_token_count
        metrics["token_count"] = count

        format_data = None
        if format_spec and complete_text:
            if DEBUG_MODE:
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
                if DEBUG_MODE:
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
                        if DEBUG_MODE:
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
                if DEBUG_MODE:
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
                                if DEBUG_MODE:
                                    logger.debug(
                                        f"Fixed JSON validation succeeded: {cleaned_json}"
                                    )
                                break
                    except:
                        continue

            if DEBUG_MODE:
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
            model_name, complete_text, metrics, format_data
        )

        if DEBUG_MODE and format_data:
            logger.debug("Created formatted response with JSON content")

        return jsonify(response), 200
