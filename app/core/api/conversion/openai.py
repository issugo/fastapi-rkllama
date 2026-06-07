from typing import List

from core.api.parameters.openai_requests import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    VisionRequest,
)
from core.api.parameters.ollama_requests import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaEmbeddingRequest,
)
from core.api.parameters.commons import Message, TextContent, ImageContent


class OpenAIConversions:
    """
    Class to handle conversions from OpenAI models to Ollama models.
    """

    @staticmethod
    def convert_chat_completion_request(
        request: ChatCompletionRequest,
    ) -> OllamaChatRequest:
        """
        Convert an OpenAI ChatCompletionRequest to an OllamaChatRequest.

        Args:
            request: The OpenAI ChatCompletionRequest to convert

        Returns:
            OllamaChatRequest: The converted Ollama chat request
        """
        # Map OpenAI options to Ollama options
        options = {}

        if request.temperature is not None:
            options["temperature"] = request.temperature

        if request.top_p is not None:
            options["top_p"] = request.top_p

        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        if request.frequency_penalty is not None:
            options["frequency_penalty"] = request.frequency_penalty

        if request.presence_penalty is not None:
            options["presence_penalty"] = request.presence_penalty

        if request.stop:
            options["stop"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        # Format handling for JSON output if specified
        format_option = None
        if (
            request.response_format
            and getattr(request.response_format, "type", None) == "json_object"
        ):
            format_option = "json"

        # Create Ollama chat request
        return OllamaChatRequest(
            model=request.model,
            messages=request.messages,
            options=options or None,
            stream=request.stream,
            format=format_option,
        )

    @staticmethod
    def convert_completion_request(request: CompletionRequest) -> OllamaGenerateRequest:
        """
        Convert an OpenAI CompletionRequest to an OllamaGenerateRequest.

        Args:
            request: The OpenAI CompletionRequest to convert

        Returns:
            OllamaGenerateRequest: The converted Ollama generate request
        """
        # Map OpenAI options to Ollama options
        options = {}

        if request.temperature is not None:
            options["temperature"] = request.temperature

        if request.top_p is not None:
            options["top_p"] = request.top_p

        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        if request.frequency_penalty is not None:
            options["frequency_penalty"] = request.frequency_penalty

        if request.presence_penalty is not None:
            options["presence_penalty"] = request.presence_penalty

        if request.stop:
            options["stop"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )

        # Extract prompt
        prompt = request.prompt
        if isinstance(prompt, list):
            # For lists, take the first item
            prompt = prompt[0] if prompt else ""

        # System prompt extraction (if in a system message format)
        system = None

        # Create Ollama generate request
        return OllamaGenerateRequest(
            model=request.model,
            prompt=prompt,
            options=options or None,
            system=system,
            stream=request.stream,
        )

    @staticmethod
    def convert_embedding_request(request: EmbeddingRequest) -> OllamaEmbeddingRequest:
        """
        Convert an OpenAI EmbeddingRequest to an OllamaEmbeddingRequest.

        Args:
            request: The OpenAI EmbeddingRequest to convert

        Returns:
            OllamaEmbeddingRequest: The converted Ollama embedding request
        """
        # Extract prompt/input
        prompt = request.input
        if isinstance(prompt, list):
            # For lists, take the first item or join them
            prompt = prompt[0] if prompt else ""

        # Create Ollama embedding request
        return OllamaEmbeddingRequest(model=request.model, prompt=prompt)

    @staticmethod
    def convert_vision_request(request: VisionRequest) -> OllamaChatRequest:
        """
        Convert an OpenAI VisionRequest to an OllamaChatRequest with images.

        Args:
            request: The OpenAI VisionRequest to convert

        Returns:
            OllamaChatRequest: The converted Ollama chat request with images
        """
        # Map OpenAI options to Ollama options
        options = {}

        if request.temperature is not None:
            options["temperature"] = request.temperature

        if request.top_p is not None:
            options["top_p"] = request.top_p

        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        # Process messages and extract images
        messages = []
        images = []

        for msg in request.messages:
            content = msg.content

            # Process content if it's a list (multimodal content)
            if isinstance(content, list):
                text_parts = []

                for item in content:
                    if isinstance(item, TextContent) or (
                        isinstance(item, dict) and item.get("type") == "text"
                    ):
                        # Extract text content
                        text = (
                            item.text
                            if isinstance(item, TextContent)
                            else item.get("text", "")
                        )
                        text_parts.append(text)
                    elif isinstance(item, ImageContent) or (
                        isinstance(item, dict) and item.get("type") == "image"
                    ):
                        # Extract image URL
                        if isinstance(item, ImageContent):
                            image_url = str(item.image_url)
                        else:
                            image_url_obj = item.get("image_url", {})
                            image_url = (
                                image_url_obj.get("url", "")
                                if isinstance(image_url_obj, dict)
                                else str(image_url_obj)
                            )

                        if image_url:
                            images.append(image_url)

                # Create a new message with only text content
                new_msg = Message(
                    role=msg.role, content="".join(text_parts), name=msg.name
                )
                messages.append(new_msg)
            else:
                # Regular text message
                messages.append(msg)

        # Create Ollama chat request with images
        ollama_request = OllamaChatRequest(
            model=request.model,
            messages=messages,
            options=options or None,
            stream=request.stream,
        )

        # Add images if any were found
        if images:
            # Note: This is a workaround as OllamaChatRequest doesn't have an 'images' field
            # In practice, we would need to extend the model or handle this at serialization
            ollama_request_dict = ollama_request.model_dump()
            ollama_request_dict["images"] = images
            return OllamaChatRequest.model_validate(ollama_request_dict)

        return ollama_request

    @staticmethod
    def extract_images_from_messages(messages: List[Message]) -> List[str]:
        """
        Extract image URLs from a list of messages.

        Args:
            messages: List of Message objects that may contain image content

        Returns:
            List[str]: List of image URLs
        """
        image_urls = []

        for msg in messages:
            content = msg.content

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, ImageContent):
                        image_urls.append(str(item.image_url))
                    elif isinstance(item, dict) and item.get("type") == "image":
                        image_url_obj = item.get("image_url", {})
                        if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                            image_urls.append(image_url_obj["url"])
                        elif isinstance(image_url_obj, str):
                            image_urls.append(image_url_obj)

        return image_urls
