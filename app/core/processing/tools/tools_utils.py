import json
import re

import logging

logger = logging.getLogger("rkllama.tools_utils")


def RawJSONDecoder(index):
    class _RawJSONDecoder(json.JSONDecoder):
        end = None

        def decode(self, s, *_):
            data, self.__class__.end = self.raw_decode(s, index)
            return data

    return _RawJSONDecoder


def extract_json_tools_from_text(s, index=0):
    while (index := s.find("{", index)) != -1:
        try:
            yield json.loads(s, cls=(decoder := RawJSONDecoder(index)))
            index = decoder.end
        except json.JSONDecodeError:
            index += 1


def get_tool_calls_generic(response):
    """Return a list of formatted function calls by the LLM in the response.
            It a generic function to search any JSON response from any LLM with the required format:
            {"name": <function_name>, "parameters": <dictionary_of_argument_name_value>}
            or
            {"name": <function_name>, "arguments": <dictionary_of_argument_name_value>}
            For example:

            { "name": "get_current_weather", "arguments": { "location": "Paris, France", "format": "celsius" }

            Qwen models use <tool_call></tool_call> tags in chat template but for example Llama3.2 doesn't. That's why this generic implementation.


            Final response of a request must something like this: (https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools)

            {
                "model": "llama3.2",
                "created_at": "2024-07-22T20:33:28.123648Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                    {
                        "function": {
                        "name": "get_current_weather",
                        "arguments": {
                            "format": "celsius",
                            "location": "Paris, FR"
                        }
                        }
                    }
                    ]
                },
                "done_reason": "stop",
                "done": true,
                "total_duration": 885095291,
                "load_duration": 3753500,
                "prompt_eval_count": 122,
                "prompt_eval_duration": 328493000,
                "eval_count": 33,
                "eval_duration": 552222000
            }


    }
    """

    logger.debug("Searching tools with generic method: get_tool_calls_generic")

    # Get all the json objects
    json_tool_list = list(extract_json_tools_from_text(response))

    # Set the required keys in json object to identify tool calls
    required_keys_for_tools_option1 = set(["name", "arguments"])  # Other like Qwen
    required_keys_for_tools_option2 = set(
        ["name", "parameters"]
    )  # Llama default chat template

    tool_calls = []
    tool_calls += [
        {"function": tool}
        for tool in json_tool_list
        if required_keys_for_tools_option1.issubset(tool.keys())
        or required_keys_for_tools_option2.issubset(tool.keys())
    ]

    # Rename the key "parameters" for "arguments" for standard
    tool_calls_renamed = []
    for tool in tool_calls:
        if "parameters" in tool["function"]:
            tool["function"]["arguments"] = tool["function"].pop("parameters")
        tool_calls_renamed.append(tool)
    return tool_calls_renamed


def get_tool_calls_standard(response):
    """Get all the tool calls indicated by the LLM in the response.
    Only work if the chat template of the LLM uses <tool_call></tool_call> tags (Like Qwen models)
    """

    logger.debug("Searching tools with standard method: get_tool_calls_standard")

    tool_calls = []
    for tools in re.findall("<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        # tool_calls += [{ "function": json.loads(tool) } for tool in tools.split('\n') if tool]
        # To make more smartt LLM in case LLM response inside <tool_call><t/ool_call> but with an extra word.
        # For example: <tool_call> Output : {"name": <function_name>, "arguments": <dictionary_of_argument_name_value>} </tool_call>
        tool_calls += get_tool_calls_generic(tools)
    return tool_calls


def get_tool_calls(response):
    """Get all the tool calls indicated by the LLM in the response"""

    # We try the standard form first
    tool_calls = get_tool_calls_standard(response)

    if not tool_calls:
        # No standard format tool call found. Search for more generic way
        tool_calls = get_tool_calls_generic(response)

    return tool_calls
