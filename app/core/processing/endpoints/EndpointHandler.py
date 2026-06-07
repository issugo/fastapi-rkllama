import time
from typing import Union, Dict, Any, Tuple, Optional, List

from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from abc import ABC

from core.api.parameters import Message
from core.backends.GlobalState import GLOBAL_STATE
from core.config.RKLLAMAConfig import RKLLAMASettings
from core.model.ModelFile import ModelFile

settings: RKLLAMASettings | None = None
DEBUG_MODE: bool | None = None


class EndpointMetrics(BaseModel):
    total: int
    prompt_eval: int
    eval: int
    load: int
    prompt_tokens: Optional[int] = Field(default=None)
    token_count: Optional[int] = Field(default=None)


class EndpointHandler(ABC):
    """Base class for endpoint handlers with common functionality"""

    @classmethod
    def prepare_prompt(
        cls,
        modelfile: ModelFile,
        messages: List[Message],
        system="",
        tools=None,
        enable_thinking=False,
    ) -> Tuple[Any, Union[list[int], Dict], int]:
        """Prepare prompt with proper system handling"""

        ## TODO: tokenizer is downloaded when get model from internet
        ## so use modelfile
        tokenizer = AutoTokenizer.from_pretrained(
            ## TODO: get loaded model from parameter
            GLOBAL_STATE.loaded_model_hfpath,
            trust_remote_code=True,
        )
        supports_system_role = (
            "raise_exception('System role not supported')"
            not in tokenizer.chat_template
        )

        # print(messages)
        # print("-" * 50)
        # prepared_messages = []
        # for message in messages:
        #    all_contents = ""
        #    for content in message.get("content", []):
        #        if isinstance(content, str):
        #            content = content.strip()
        #            if content:
        #                all_contents = all_contents + "\n" +message["content"]
        #        elif isinstance(content, dict):
        #                if "text" in content:
        #                    content["text"] = content["text"].strip()
        #                    if content["text"]:
        #                        all_contents = all_contents + "\n" + content["text"]
        #    prepared_messages.append({"role": message["role"], "content": all_contents})

        flat_messages = [
            {"role": message.role.value, "content": str(message.content)}
            for message in messages
        ]
        if system and supports_system_role:
            # prompt_messages = [{"role": "system", "content": system}] + prepared_messages #messages
            prompt_messages = [{"role": "system", "content": system}] + flat_messages
        else:
            prompt_messages = flat_messages

        prompt_tokens: Union[list[int], Dict] = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        return tokenizer, prompt_tokens, len(prompt_tokens)

    @classmethod
    def calculate_durations(
        cls, start_time, prompt_eval_time, current_time=None
    ) -> EndpointMetrics:
        """Calculate duration metrics for responses"""
        if not current_time:
            current_time = time.time()

        total_duration = current_time - start_time

        if prompt_eval_time is None:
            prompt_eval_time = start_time + (total_duration * 0.1)

        prompt_eval_duration = prompt_eval_time - start_time
        eval_duration = current_time - prompt_eval_time

        return EndpointMetrics(
            total=int(total_duration * 1_000_000_000),
            prompt_eval=int(prompt_eval_duration * 1_000_000_000),
            eval=int(eval_duration * 1_000_000_000),
            load=int(0.1 * 1_000_000_000),
        )

    @classmethod
    def settings(cls) -> RKLLAMASettings:
        global settings
        if settings is None:
            from core.config import config_utils

            settings = config_utils.get_settings()
        return settings

    @property
    def DEBUG_MODE(cls) -> bool:
        global DEBUG_MODE
        if DEBUG_MODE is None:
            DEBUG_MODE = cls.settings.is_debug_mode()
        return DEBUG_MODE
