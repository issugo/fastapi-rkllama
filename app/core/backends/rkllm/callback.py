import sys
import logging

from core.model.Model import Model
import core.backends.GlobalState
from core.backends.rkllm.classes import LLMCallState

# Get logger for this module
logger = logging.getLogger("core.rkllm.callback")

split_byte_data = bytes(b"")

# Definir la fonction de rappel
def callback_impl(result, userdata, status):
    global split_byte_data

    rkllm_model: core.model.Model.Model = core.backends.GlobalState.GLOBAL_STATE.backend

    if status == LLMCallState.RKLLM_RUN_FINISH:
        rkllm_model.shared_data.global_status = status
        logger.info("Generation completed")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_ERROR:
        rkllm_model.shared_data.global_status = status
        logger.error("erreur d'execution")
        sys.stdout.flush()
    elif status == LLMCallState.RKLLM_RUN_NORMAL:
        # Sauvegarder le texte du token de sortie et l'status d'execution de RKLLM
        rkllm_model.shared_data.global_status = status
        # Check if result or result.contents or result.contents.text is None
        try:
            # Add defensive checks to prevent None concatenation
            if result and result.contents and result.contents.text:
                text_bytes = result.contents.text
                if not isinstance(text_bytes, bytes):
                    # If not bytes, try to convert or use empty bytes
                    try:
                        text_bytes = bytes(text_bytes)
                    except Exception as conv_error:
                        logger.error(f"Error converting to bytes: {str(conv_error)}")
                        text_bytes = b""

                # Now safely concatenate
                try:
                    decoded_text = (split_byte_data + text_bytes).decode('utf-8')
                    rkllm_model.shared_data.global_text.append(decoded_text)
                    logger.debug(f"Token: {decoded_text}")
                    split_byte_data = bytes(b"")
                except UnicodeDecodeError as decode_error:
                    # Handle incomplete UTF-8 sequences
                    logger.debug(f"Incomplete UTF-8 sequence received")
                    split_byte_data += text_bytes
            else:
                # Handle case where text is None
                if split_byte_data:
                    try:
                        # Try to decode any accumulated bytes
                        decoded_text = split_byte_data.decode('utf-8')
                        rkllm_model.shared_data.global_text.append(decoded_text)
                        logger.debug(f"Token from accumulated bytes: {decoded_text}")
                        split_byte_data = bytes(b"")
                    except UnicodeDecodeError as decode_error:
                        # Still incomplete, keep for next time
                        logger.debug(f"Still incomplete UTF-8 sequence")
                        pass
        except Exception as e:
            logger.error(f"Error processing callback: {str(e)}")
            
        sys.stdout.flush()