import json
import threading
import time
from threading import Thread

from fastapi.responses import JSONResponse, StreamingResponse

import core.config.config_utils
from core import config
import logging
from core.config.config_utils import is_debug_mode
from core.model.Model import ModelSharedData
from core.model.ModelFile import ModelFile
from core.api.parameters import Message, Role
from core.processing.APIHandler import DataFormat, Counters, APIHandler, SharedData
from core.processing.endpoints.ChatEndpointHandler import ChatEndpointHandler
from core.processing.endpoints.GenerateEndpointHandler import GenerateEndpointHandler
from core.processing.format_spec.formatting import create_format_instruction, validate_format_response
from core.backends.rkllm.rkllm_backend import RKLLMBackend

logger = logging.getLogger("core.processing.process")

# Get DEBUG_MODE from config instead of the environment variable
DEBUG_MODE = is_debug_mode()

import os
from typing import Optional, List
from transformers import AutoTokenizer
from dotenv import load_dotenv

is_locked: bool = False

def load_tokenizer(model_file: ModelFile) -> Optional[AutoTokenizer]:
    model_id: str = model_file.huggingface_path
    # Load environment variables from Modelfile
    load_dotenv(model_file.file, override=True)

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

async def get_messages(data: dict | None, data_format: DataFormat) -> List[Message]:
    # Get chat history from JSON request
    messages: List[Message] = list(map(lambda json_message: Message(**json.loads(json_message)), data["messages"]))

    # Create format instructions
    if data_format.format_spec:
        format_instruction = create_format_instruction(data_format.format_spec)
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
    return messages

async def rkllm_request(rkllm_model: RKLLMBackend, model_shared_data: ModelSharedData, model_file: ModelFile, usage_lock: threading.Lock, handler: APIHandler, data: dict = None) -> JSONResponse | StreamingResponse:
    """
    Process a request to the language model

    Args:
        rkllm_model: The RKLLM language model instance
        model_shared_data: The ModelSharedData instance
        model_file: The ModelFile associated with rkllm_model
        data: Optional custom request object that mimics request

    Returns:
        FatsAPI response with generated text
    """
    try:
        # Put the server in a locked state
        global is_locked
        is_locked = True

        if data and "messages" in data:
            # Extract format parameters
            data_format: DataFormat = DataFormat(
                format_spec=core.config.config_utils.get("format"),
                format_options=core.config.config_utils.get("options", {}))

            # Store format settings in model instance for reference
            if rkllm_model:
                rkllm_model.format_schema = data_format.format_spec
                rkllm_model.format_type = (
                    data_format.format_spec.get("type", "")
                    if isinstance(data_format.format_spec, dict)
                    else data_format.format_spec
                )
                rkllm_model.format_options = data_format.format_options

            # Reset global variables
            model_shared_data.global_status = -1

            messages = await get_messages(data, data_format)

            # Setup tokenizer
            tokenizer = load_tokenizer(model_file)

            supports_system_role = (
                "raise_exception('System role not supported')"
                not in tokenizer.chat_template
            )

            if (model_file.system_prompt != "") and supports_system_role:
                prompt = [Message(role= Role.SYSTEM, content= model_file.system_prompt)] + messages
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


            counters: Counters = Counters()
            shared_data: SharedData = SharedData(data_format)
            response = handler.new_response()

            # Create the inference thread
            model_thread: Thread = threading.Thread(
                target=rkllm_model.run, args=(prompt,)
            )

            if ("stream" not in data.keys()) or ("stream" in data.keys() and data["stream"] == True):


                # Return appropriate streaming response based on request type
                ## return Response(generate(), content_type='application/x-ndjson' if is_ollama_request else 'text/plain')
                content_type = handler.response_content_type
                logger.info("Returning streaming response")
                return StreamingResponse(
                    handler.generate(counters=counters, shared_data=shared_data, response=response, model_thread=model_thread, model_shared_data=model_shared_data),
                    headers={"Content-Type": content_type}
                )

            # For non-streaming responses
            else:
                try:
                    model_thread.start()
                    logger.info("Inference thread started")
                except Exception as e:
                    logger.error(f"Error starting thread: {e}")

                # Wait for model to finish
                thread_model_finished = False
                counters.count = 0
                counters.start = time.time()
                counters.prompt_eval_end_time = (
                    None  # Will store time when first token is generated
                )
                counters.complete_text = ""
                counters.first_token_generated = False

                while not thread_model_finished:
                    while len(model_shared_data.global_text) > 0:
                        counters.count += 1
                        token = model_shared_data.global_text.pop(0)

                        # Mark the time when first token is generated (end of prompt evaluation)
                        if not counters.first_token_generated:
                            counters.first_token_generated = True
                            counters.prompt_eval_end_time = time.time()

                        counters.complete_text += token
                        time.sleep(0.005)

                        model_thread.join(timeout=0.005)
                    thread_model_finished = not model_thread.is_alive()

                end_time = time.time()
                counters.total_duration = end_time - counters.start

                # Calculate the various duration metrics
                if counters.prompt_eval_end_time is None:
                    # If no tokens were generated, use 10% of total time as estimate
                    counters.prompt_eval_end_time = counters.start + (counters.total_duration * 0.1)

                counters.prompt_eval_duration = (
                    counters.prompt_eval_end_time - counters.start
                )  # Time spent evaluating prompt
                counters.eval_duration = (
                    end_time - counters.prompt_eval_end_time
                )  # Time spent generating tokens
                counters.load_duration = 0.1  # Fixed 100ms in seconds

                # Handle format validation for completed response
                ## if format_spec and complete_text:
                if counters.complete_text:
                    # Updated to unpack the additional cleaned_json return value
                    shared_data.success, shared_data.parsed_data, shared_data.error, shared_data.cleaned_json = (
                        validate_format_response(counters.complete_text, data_format.format_spec)
                    )
                    logger.debug(f"Format validation: success={shared_data.success}, error={shared_data.error}")

                # Prepare appropriate response based on request type
                formatted_response = handler.format_response(
                    response=response,
                    prompt=prompt,
                    usage_prompt_tokens=len(prompt),
                    counters=counters,
                    shared_data=shared_data)

                ## return jsonify(ollama_response), 200
                usage_lock.release()
                return JSONResponse(formatted_response, status_code=200)

        else:
            ## return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
            usage_lock.release()
            return JSONResponse(
                {"status": "error", "message": "Invalid JSON data!"}, status_code=400
            )
    except Exception as e:
        # No need to release the lock here as it should be handled by the calling function
        logger.error(f"Request processing error: {e}", exc_info=True)
        usage_lock.release()
        is_locked = False
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON data!"}, status_code=400
        )


def process_ollama_chat_request(
    modele_rkllm,
    model_name,
    messages,
    system="",
    stream=True,
    format_spec=None,
    options=None,
):
    """Process /api/chat request with correct format"""
    return ChatEndpointHandler.handle_request(
        modele_rkllm=modele_rkllm,
        model_name=model_name,
        messages=messages,
        system=system,
        stream=stream,
        format_spec=format_spec,
        options=options,
    )


def process_ollama_generate_request(
    modele_rkllm,
    model_name,
    prompt,
    system="",
    stream=True,
    format_spec=None,
    options=None,
):
    """Process /api/generate request with correct format"""
    return GenerateEndpointHandler.handle_request(
        modele_rkllm=modele_rkllm,
        model_name=model_name,
        prompt=prompt,
        system=system,
        stream=stream,
        format_spec=format_spec,
        options=options,
    )
