import json
from typing import List

from core.api.parameters import Message
from core.processing.APIHandler import DataFormat
from core.processing.format_spec.formatting import create_format_instruction
from core.processing.process import DEBUG_MODE, logger


async def get_messages(data: dict | None, data_format: DataFormat) -> List[Message]:
    # Get chat history from JSON request
    messages: List[Message] = list(
        map(lambda json_message: Message(**json.loads(json_message)), data["messages"])
    )

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
                    logger.debug(f"Added format instruction: {format_instruction}")
    return messages
