import threading
import time
import json

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

import core.rkllm.GlobalState
from core import config
import src.variables as variables
import datetime
import logging
from core.config import is_debug_mode  # Import the config module
from core.processing.formatting import create_format_instruction, validate_format_response
from app.core.rkllm.GlobalState import GLOBAL_STATE

logger = logging.getLogger("rkllama.process")

# Get DEBUG_MODE from config instead of environment variable
DEBUG_MODE = is_debug_mode()

import os
from typing import Optional
from transformers import AutoTokenizer
import urllib.parse
from dotenv import load_dotenv


def load_tokenizer(modelfile: str, model_id: str) -> Optional[AutoTokenizer]:
    # Load environment variables from Modelfile
    load_dotenv(modelfile, override=True)

    # Retrieve custom tokenizer path
    custom_tokenizer = os.getenv("TOKENIZER")
    tokenizer = None

    if custom_tokenizer:
        # Check if the custom tokenizer path exists
        if os.path.exists(custom_tokenizer):
            try:
                # Attempt to load the custom tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    custom_tokenizer, trust_remote_code=True
                )
                logger.info(f"Loaded custom tokenizer from {custom_tokenizer}")
            except Exception as e:
                # Warn user and prepare to fallback
                logger.warning(
                    f"Could not load tokenizer from {custom_tokenizer}. Error: {str(e)}. Falling back to default tokenizer."
                )
        else:
            # Warn user if path is invalid
            logger.warning(
                f"Tokenizer path {custom_tokenizer} does not exist. Falling back to default tokenizer."
            )

    # Fallback to default AutoTokenizer if necessary
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            logger.info(f"Loaded default tokenizer for model {model_id}")
        except Exception as e:
            logger.error(
                f"Failed to load default tokenizer for {model_id}. Error: {str(e)}."
            )
            return None

    return tokenizer


async def CustomRequest(rkllm_model, modelfile, request: Request = None):
    """
    Process a request to the language model

    Args:
        rkllm_model: The language model instance
        custom_request: Optional custom request object that mimics Flask request

    Returns:
        Flask response with generated text
    """
    try:
        # Put the server in a locked state
        is_locked = True

        # Use custom_request if provided, otherwise use Flask's request
        # req = custom_request if custom_request is not None else request
        # data = req.json
        data = await request.json()

        if data and "messages" in data:
            # Extract format parameters
            format_spec = config.get("format")
            format_options = config.get("options", {})

            # Store format settings in model instance for reference
            if rkllm_model:
                rkllm_model.format_schema = format_spec
                rkllm_model.format_type = (
                    format_spec.get("type", "")
                    if isinstance(format_spec, dict)
                    else format_spec
                )
                rkllm_model.format_options = format_options

            # Reset global variables
            GLOBAL_STATE.global_status = -1

            # Define the structure of the returned response
            llmResponse = {
                "id": "rkllm_chat",
                "object": "rkllm_chat",
                "created": int(time.time()),
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0,
                    "total_tokens": 0,
                },
            }

            # Check if this is an Ollama-style request
            urlparsed_path = urllib.parse.urlparse(str(request.url)).path
            ## is_ollama_request = req.path.startswith('/api/')
            is_ollama_request = urlparsed_path.startswith("/api/")

            # Get chat history from JSON request
            messages = data["messages"]

            # Create format instructions
            if format_spec:
                format_instruction = create_format_instruction(format_spec)
                if format_instruction:
                    # Find the last user message and append format instructions
                    last_user_msg_idx = -1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            last_user_msg_idx = i
                            break

                    if last_user_msg_idx >= 0:
                        original_content = messages[last_user_msg_idx]["content"]
                        messages[last_user_msg_idx]["content"] = (
                            original_content + format_instruction
                        )
                        if DEBUG_MODE:
                            logger.debug(
                                f"Added format instruction: {format_instruction}"
                            )

            # Setup tokenizer
            tokenizer = load_tokenizer(modelfile, GLOBAL_STATE.loaded_model_hfpath)

            supports_system_role = (
                "raise_exception('System role not supported')"
                not in tokenizer.chat_template
            )

            if variables.system and supports_system_role:
                prompt = [{"role": "system", "content": variables.system}] + messages
            else:
                prompt = messages

            for i in range(1, len(prompt)):
                if prompt[i]["role"] == prompt[i - 1]["role"]:
                    raise ValueError(
                        "Roles must alternate between 'user' and 'assistant'."
                    )

            # Set up chat template
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
            llmResponse["usage"]["prompt_tokens"] = llmResponse["usage"][
                "total_tokens"
            ] = len(prompt)

            sortie_rkllm = ""

            if "stream" in data.keys() and data["stream"] == True:

                def generate():
                    count = 0
                    start = time.time()
                    prompt_eval_end_time = None
                    final_message_sent = False  # Track if we've sent the final message

                    # Initialize accumulated text for JSON format validation
                    complete_text = ""
                    tokens_since_last_response = (
                        0  # Track tokens since last response sent
                    )

                    thread_modele = threading.Thread(
                        target=rkllm_model.run, args=(prompt,)
                    )
                    thread_modele.start()

                    thread_model_finished = False

                    while not thread_model_finished:
                        logger.debug("rkllm thread running")

                        while len(GLOBAL_STATE.global_text) > 0:
                            current_token = GLOBAL_STATE.global_text.pop(0)
                            llmResponse["choices"] = [
                                {
                                    "role": "assistant",
                                    "content": current_token,
                                    "logprobs": None,
                                    "finish_reason": "stop"
                                    if GLOBAL_STATE.global_status == 1
                                    else None,
                                }
                            ]
                            llmResponse["usage"]["completion_tokens"] = count
                            llmResponse["usage"]["total_tokens"] += 1

                            # Process format in the final chunk
                            if variables.global_status == 1 and format_spec:
                                success, parsed_data, error, cleaned_json = (
                                    validate_format_response(complete_text, format_spec)
                                )
                                if success and parsed_data:
                                    llmResponse["choices"][0]["format"] = format_spec
                                    llmResponse["choices"][0]["parsed"] = parsed_data

                            # Send the response
                            yield f"{json.dumps(llmResponse)}\n\n"

                        logger.debug("sleeping")
                        time.sleep(0.005)
                        logger.debug("joining")
                        thread_modele.join(timeout=0.005)
                        thread_model_finished = not thread_modele.is_alive()

                    logger.info("rkllm thread finished")

                # Return appropriate streaming response based on request type
                ## return Response(generate(), content_type='application/x-ndjson' if is_ollama_request else 'text/plain')
                content_type = (
                    "application/x-ndjson" if is_ollama_request else "text/plain"
                )
                logger.info("Returning streaming response")
                return StreamingResponse(
                    generate(), headers={"Content-Type": content_type}
                )

            # For non-streaming responses
            else:
                # Create inference thread
                thread_modele = threading.Thread(
                    target=rkllm_model.run, args=(prompt,)
                )
                try:
                    thread_modele.start()
                    logger.info("Inference thread started")
                except Exception as e:
                    logger.error(f"Error starting thread: {e}")

                # Wait for model to finish
                thread_model_finished = False
                count = 0
                start = time.time()
                prompt_eval_end_time = (
                    None  # Will store time when first token is generated
                )
                complete_text = ""
                first_token_generated = False

                while not thread_model_finished:
                    while len(GLOBAL_STATE.global_text) > 0:
                        count += 1
                        token = GLOBAL_STATE.global_text.pop(0)

                        # Mark the time when first token is generated (end of prompt evaluation)
                        if not first_token_generated:
                            first_token_generated = True
                            prompt_eval_end_time = time.time()

                        complete_text += token
                        time.sleep(0.005)

                        thread_modele.join(timeout=0.005)
                    thread_model_finished = not thread_modele.is_alive()

                end_time = time.time()
                total_duration = end_time - start

                # Calculate the various duration metrics
                if prompt_eval_end_time is None:
                    # If no tokens were generated, use 10% of total time as estimate
                    prompt_eval_end_time = start + (total_duration * 0.1)

                prompt_eval_duration = (
                    prompt_eval_end_time - start
                )  # Time spent evaluating prompt
                eval_duration = (
                    end_time - prompt_eval_end_time
                )  # Time spent generating tokens
                load_duration = 0.1  # Fixed 100ms in seconds

                # Handle format validation for completed response
                ## if format_spec and complete_text:
                if complete_text:
                    # Updated to unpack the additional cleaned_json return value
                    success, parsed_data, error, cleaned_json = (
                        validate_format_response(complete_text, format_spec)
                    )
                    logger.debug(f"Format validation: success={success}, error={error}")

                # Prepare appropriate response based on request type
                if is_ollama_request:
                    ollama_response = {
                        "model": GLOBAL_STATE.loaded_model_hfpath,
                        "created_at": datetime.datetime.now().strftime(
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
                        "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                        "prompt_eval_duration": int(
                            prompt_eval_duration * 1_000_000_000
                        ),
                        "eval_count": count,
                        "eval_duration": int(eval_duration * 1_000_000_000),
                    }

                    ## return jsonify(ollama_response), 200
                    variables.verrou.release()
                    return JSONResponse(ollama_response, status_code=200)
                else:
                    # Standard RKLLAMA API response
                    llmResponse["choices"] = [
                        {
                            "role": "assistant",
                            # Use only the clean JSON text if available
                            "content": cleaned_json
                            if success and cleaned_json
                            else complete_text,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ]

                    # Add format information if available
                    if success and parsed_data:
                        llmResponse["choices"][0]["format"] = format_spec
                        llmResponse["choices"][0]["parsed"] = parsed_data

                    # Update token counts
                    llmResponse["usage"]["completion_tokens"] = count
                    llmResponse["usage"]["total_tokens"] = (
                        llmResponse["usage"]["prompt_tokens"] + count
                    )

                    # Calculate tokens per second if we have meaningful duration
                    if eval_duration > 0:
                        llmResponse["usage"]["tokens_per_second"] = round(
                            count / eval_duration, 2
                        )

                    ## return jsonify(llmResponse), 200
                    variables.verrou.release()
                    return JSONResponse(llmResponse, status_code=200)

        else:
            ## return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
            variables.verrou.release()
            return JSONResponse(
                {"status": "error", "message": "Invalid JSON data!"}, status_code=400
            )
    except Exception as e:
        # No need to relese the lock here as it should be handled by the calling function
        logger.error(f"Request processing error: {e}", exc_info=True)
        variables.verrou.release()
        is_locked = False
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON data!"}, status_code=400
        )
