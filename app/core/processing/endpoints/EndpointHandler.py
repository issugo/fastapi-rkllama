import time

from transformers import AutoTokenizer

from core.backends.GlobalState import GLOBAL_STATE


class EndpointHandler():
    """Base class for endpoint handlers with common functionality"""

    @classmethod
    def prepare_prompt(cls, messages, system="", tools=None, enable_thinking=False):
        """Prepare prompt with proper system handling"""
        tokenizer = AutoTokenizer.from_pretrained(
            GLOBAL_STATE.loaded_model_hfpath, trust_remote_code=True
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

        if system and supports_system_role:
            # prompt_messages = [{"role": "system", "content": system}] + prepared_messages #messages
            prompt_messages = [{"role": "system", "content": system}] + messages
        else:
            prompt_messages = messages

        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        return tokenizer, prompt_tokens, len(prompt_tokens)

    @classmethod
    def calculate_durations(cls, start_time, prompt_eval_time, current_time=None):
        """Calculate duration metrics for responses"""
        if not current_time:
            current_time = time.time()

        total_duration = current_time - start_time

        if prompt_eval_time is None:
            prompt_eval_time = start_time + (total_duration * 0.1)

        prompt_eval_duration = prompt_eval_time - start_time
        eval_duration = current_time - prompt_eval_time

        return {
            "total": int(total_duration * 1_000_000_000),
            "prompt_eval": int(prompt_eval_duration * 1_000_000_000),
            "eval": int(eval_duration * 1_000_000_000),
            "load": int(0.1 * 1_000_000_000),
        }
