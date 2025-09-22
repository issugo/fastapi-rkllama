import json
from typing import Optional, List, Dict, Any

from pydantic import BaseModel
from pydantic_core import from_json

from core.model import OllamaManifest
from core.model.ModelType import ModelType

"""
devstral:latest (from) sample for OllamaModelConfig
config_data = {
    "model_format": "gguf",
    "model_family": "llama",
    "model_families": ["llama"],
    "model_type": "23.6B",
    "file_type": "Q4_K_M",
    "architecture": "amd64",
    "os": "linux",
    "rootfs": {
        "type": "layers",
        "diff_ids": [
            "sha256:b3a2c9a8fef9be8d2ef951aecca36a36b9ea0b70abe9359eab4315bf4cd9be01",
            "sha256:6db27cd4e277c91264572b9c899c1980daa8dea11e902f0070a6f4763f3d13c8",
            "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            "sha256:5725afc40acd80cbeefba61e41cf50eb7924f6ed2fe6aec2dc6fa0e9f2c396d1"
        ]
    }
}

"""


class OllamaModelDetails(BaseModel):
    model_format: Optional[str] = None # ex: gguf
    model_family: Optional[str] = None # ex: llama, see


class OllamaRootfs(BaseModel):
    type: str
    diff_ids: List[str]  # list of sha256 to model, license & system files


class ModelDetails(OllamaModelDetails):
    parameter_size: str  # ex: 3B
    quantization_level: str


class ModelInfo(BaseModel):
    name: str  # Use simplified name like qwen:3b
    model: str  # Match Ollama's format
    modified_at: str
    size: int
    digest: str = ""  # Ollama field (sha256 value) (not used but included for compatibility)
    details: ModelDetails
    model_type: ModelType


class OllamaModelInfo(OllamaModelDetails):
    model_families: List[str]
    model_type: str  # parameter size
    file_type: str  # quantization level
    architecture: str  # ex: amd64, from processor
    os: Optional[str] = "Linux"
    rootfs: OllamaRootfs

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            return cls.model_validate_json(**from_json(json.load(f)))

    def save(self, file_path: str):
        """ write in <MODELS>/blobs/sha256-<DIGEST>"""
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))


class HFModelConfig(BaseModel):
    architectures: List[str]
    model_type: str
    tokenizer_config: Dict[str, Optional[str]]
    chat_template_jinja: Optional[str] = None


class HFCardData(BaseModel):
    base_model: List[str]
    tags: List[str]
    params: int


class HFSibling(BaseModel):
    rfilename: str


"""
hf_modelinfo_sample: dict = {'_id': '6832397972750614b899eba5',
                             'id': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                             'private': False,
                             'tags': ['qwen3', 'unsloth', 'base_model:Qwen/Qwen3-1.7B',
                                      'base_model:finetune:Qwen/Qwen3-1.7B', 'region:us', 'rockchip', 'rk3588'],
                             'downloads': 3,
                             'likes': 1,
                             'modelId': 'dulimov/Qwen3-1.7B-rk3588-1.2.1-unsloth-16k',
                             'author': 'dulimov',
                             'sha': 'd2dfbce3a30f60b584734d4e06cf145ace54a578',
                             'lastModified': '2025-05-24T21:37:33.000Z',
                             'gated': False,
                             'disabled': False,
                             'model-index': None,
                             'config': {'architectures': ['Qwen3ForCausalLM'], 'model_type': 'qwen3',
                                        'tokenizer_config': {'bos_token': None, 'eos_token': '<|im_end|>',
                                                             'pad_token': '<|vision_pad|>', 'unk_token': None},
                                        'chat_template_jinja': '{%- if tools %}\n {{- \'<|im_start|>system\\n\' }}\n {%- if messages[0].role == \'system\' %}\n {{- messages[0].content + \'\\n\\n\' }}\n {%- endif %}\n {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n {%- for tool in tools %}\n {{- "\\n" }}\n {{- tool | tojson }}\n {%- endfor %}\n {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n {%- if messages[0].role == \'system\' %}\n {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for forward_message in messages %}\n {%- set index = (messages|length - 1) - loop.index0 %}\n {%- set message = messages[index] %}\n {%- set current_content = message.content if message.content is not none else \'\' %}\n {%- set tool_start = \'<tool_response>\' %}\n {%- set tool_start_length = tool_start|length %}\n {%- set start_of_message = current_content[:tool_start_length] %}\n {%- set tool_end = \'</tool_response>\' %}\n {%- set tool_end_length = tool_end|length %}\n {%- set start_pos = (current_content|length) - tool_end_length %}\n {%- if start_pos < 0 %}\n {%- set start_pos = 0 %}\n {%- endif %}\n {%- set end_of_message = current_content[start_pos:] %}\n {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}\n {%- set ns.multi_step_tool = false %}\n {%- set ns.last_query_index = index %}\n {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}\n {%- elif message.role == "assistant" %}\n {%- set content = message.content %}\n {%- set reasoning_content = \'\' %}\n {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n {%- set reasoning_content = message.reasoning_content %}\n {%- else %}\n {%- if \'</think>\' in message.content %}\n {%- set content = (message.content.split(\'</think>\')|last).lstrip(\'\\n\') %}\n {%- set reasoning_content = (message.content.split(\'</think>\')|first).rstrip(\'\\n\') %}\n {%- set reasoning_content = (reasoning_content.split(\'<think>\')|last).lstrip(\'\\n\') %}\n {%- endif %}\n {%- endif %}\n {%- if loop.index0 > ns.last_query_index %}\n {%- if loop.last or (not loop.last and reasoning_content) %}\n {{- \'<|im_start|>\' + message.role + \'\\n<think>\\n\' + reasoning_content.strip(\'\\n\') + \'\\n</think>\\n\\n\' + content.lstrip(\'\\n\') }}\n {%- else %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n {%- endif %}\n {%- else %}\n {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n {%- endif %}\n {%- if message.tool_calls %}\n {%- for tool_call in message.tool_calls %}\n {%- if (loop.first and content) or (not loop.first) %}\n {{- \'\\n\' }}\n {%- endif %}\n {%- if tool_call.function %}\n {%- set tool_call = tool_call.function %}\n {%- endif %}\n {{- \'<tool_call>\\n{"name": "\' }}\n {{- tool_call.name }}\n {{- \'", "arguments": \' }}\n {%- if tool_call.arguments is string %}\n {{- tool_call.arguments }}\n {%- else %}\n {{- tool_call.arguments | tojson }}\n {%- endif %}\n {{- \'}\\n</tool_call>\' }}\n {%- endfor %}\n {%- endif %}\n {{- \'<|im_end|>\\n\' }}\n {%- elif message.role == "tool" %}\n {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n {{- \'<|im_start|>user\' }}\n {%- endif %}\n {{- \'\\n<tool_response>\\n\' }}\n {{- message.content }}\n {{- \'\\n</tool_response>\' }}\n {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n {{- \'<|im_end|>\\n\' }}\n {%- endif %}\n {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n {{- \'<|im_start|>assistant\\n\' }}\n {%- if enable_thinking is defined and enable_thinking is false %}\n {{- \'<think>\\n\\n</think>\\n\\n\' }}\n {%- endif %}\n{%- endif %}'},
                             'cardData': {'base_model': ['Qwen/Qwen3-1.7B'], 'tags': ['unsloth'], 'params': 1700000000},
                             'siblings': [{'rfilename': '.gitattributes'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g128-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g256-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-0-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.0.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-0.5.rkllm'},
                                          {'rfilename': 'Qwen3-1.7B-rk3588-w8a8_g512-opt-1-hybrid-ratio-1.0.rkllm'},
                                          {'rfilename': 'README.md'}, {'rfilename': 'added_tokens.json'},
                                          {'rfilename': 'chat_template.jinja'}, {'rfilename': 'config.json'},
                                          {'rfilename': 'generation_config.json'}, {'rfilename': 'merges.txt'},
                                          {'rfilename': 'special_tokens_map.json'}, {'rfilename': 'tokenizer.json'},
                                          {'rfilename': 'tokenizer_config.json'}, {'rfilename': 'vocab.json'}],
                             'spaces': [],
                             'createdAt': '2025-05-24T21:26:17.000Z',
                             'usedStorage': 17308806896,
                             'languages': ['en']}
"""


class HFModelInfo(BaseModel):
    _id: str
    id: str
    private: bool
    tags: List[str]
    downloads: int
    likes: int
    modelId: str
    author: str
    sha: str
    lastModified: str
    gated: bool
    disabled: bool
    model_index: Optional[Any] = None
    config: HFModelConfig
    cardData: HFCardData
    siblings: List[HFSibling]
    spaces: List[Any] = []
    createdAt: str
    usedStorage: int
    languages: List[str]

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            return cls.model_validate_json(**from_json(json.load(f)))

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
