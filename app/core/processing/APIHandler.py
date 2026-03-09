import time
from threading import Thread
from typing import List
from abc import ABC, abstractmethod

from core.model.Model import ModelSharedData
from core.api.parameters import Message


class DataFormat:
    def __init__(self, format_spec, format_options):
        self.format_spec = format_spec
        self.format_options = format_options


class SharedData:
    def __init__(self, data_format: DataFormat):
        self.data_format = data_format
        self.success = False
        self.parsed_data = []
        self.error = None
        self.cleaned_json: str = ""


class Counters:
    def __init__(self):
        self.created_time: int = int(time.time())
        self.start = 0
        self.prompt_eval_duration = 0
        self.eval_duration = 0
        self.load_duration = 0
        self.total_duration = 0
        self.prompt_eval_end_time = None
        self.first_token_generated = False
        self.count = 0
        self.final_message_sent = False
        self.tokens_since_last_response = 0
        self.complete_text = ""

class APIHandler(ABC):
    def __init__(self, response_content_type: str):
        self.response_content_type = response_content_type

    @abstractmethod
    def generate(self, counters: Counters, shared_data: SharedData, response, model_thread: Thread, model_shared_data: ModelSharedData):
        pass

    @abstractmethod
    def get_messages(data: dict | None, data_format: DataFormat) -> List[Message]:
        pass

    @abstractmethod
    def new_response(self):
        pass

    @abstractmethod
    def format_response(self, response, prompt: str, usage_prompt_tokens: int, counters: Counters, shared_data: SharedData):
        pass
