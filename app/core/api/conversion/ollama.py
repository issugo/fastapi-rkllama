from typing import List

from core.api.parameters.ollama_requests import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaEmbeddingRequest
)
from core.api.parameters.openai_requests import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    VisionRequest
)
from core.api.parameters.openai_commons import OpenAIJSONResponseFormat
from core.api.parameters.commons import Message, TextContent, ImageContent


class OllamaConversions:
    """
    Class to handle conversions from Ollama models to OpenAI models.
    """

    @staticmethod
    def convert_chat_request(request: OllamaChatRequest) -> ChatCompletionRequest:
        """
        Convert an Ollama ChatRequest to an OpenAI ChatCompletionRequest.

        Args:
            request: The Ollama ChatRequest to convert

        Returns:
            ChatCompletionRequest: The converted OpenAI chat completion request
        """
        # Extract options from Ollama request
        options = request.options or {}

        # Handle response format if specified
        response_format = None
        if request.format == "json":
            response_format = OpenAIJSONResponseFormat(type="json_object")

        # Map Ollama parameters to OpenAI parameters
        temperature = options.get("temperature")
        top_p = options.get("top_p")
        max_tokens = options.get("num_predict")
        frequency_penalty = options.get("frequency_penalty")
        presence_penalty = options.get("presence_penalty")
        stop = options.get("stop")

        # Process images if they exist in the Ollama request
        # Note: This assumes images might be added to the request object
        # as seen in the OpenAIConversions class
        images = getattr(request, "images", None)
        messages = request.messages

        # If images are present, we need to convert to multimodal format
        if images and isinstance(images, list):
            new_messages = []

            for msg in messages:
                if msg.role == "user" and isinstance(msg.content, str):
                    # For the last user message, add image content
                    content_items = [
                        TextContent(type="text", text=msg.content)
                    ]

                    # Add image content
                    for image_url in images:
                        content_items.append(
                            ImageContent(type="image", image_url=image_url)
                        )

                    # Create new message with multimodal content
                    new_msg = Message(
                        role=msg.role,
                        content=content_items,
                        name=msg.name
                    )
                    new_messages.append(new_msg)
                else:
                    # Keep other messages as is
                    new_messages.append(msg)

            messages = new_messages

        # Create OpenAI chat completion request
        return ChatCompletionRequest(
            model=request.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=request.stream,
            response_format=response_format
        )

    @staticmethod
    def convert_generate_request(request: OllamaGenerateRequest) -> CompletionRequest:
        """
        Convert an Ollama GenerateRequest to an OpenAI CompletionRequest.

        Args:
            request: The Ollama GenerateRequest to convert

        Returns:
            CompletionRequest: The converted OpenAI completion request
        """
        # Extract options from Ollama request
        options = request.options or {}

        # Map Ollama parameters to OpenAI parameters
        temperature = options.get("temperature")
        top_p = options.get("top_p")
        max_tokens = options.get("num_predict")
        frequency_penalty = options.get("frequency_penalty")
        presence_penalty = options.get("presence_penalty")
        stop = options.get("stop")

        # Handle system prompt if present
        prompt = request.prompt
        if request.system:
            # In OpenAI, we might want to prepend the system prompt to the user prompt
            # But CompletionRequest doesn't have a system field, so we combine them
            prompt = f"{request.system}\n\n{prompt}"

        # Create OpenAI completion request
        return CompletionRequest(
            model=request.model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=request.stream
        )

    @staticmethod
    def convert_embedding_request(request: OllamaEmbeddingRequest) -> EmbeddingRequest:
        """
        Convert an Ollama EmbeddingRequest to an OpenAI EmbeddingRequest.

        Args:
            request: The Ollama EmbeddingRequest to convert

        Returns:
            EmbeddingRequest: The converted OpenAI embedding request
        """
        # Simply map the prompt to input
        return EmbeddingRequest(
            model=request.model,
            input=request.prompt
        )

    @staticmethod
    def convert_generate_with_images_to_vision(
        request: OllamaGenerateRequest, 
        images: List[str]
    ) -> VisionRequest:
        """
        Convert an Ollama GenerateRequest with images to an OpenAI VisionRequest.

        Args:
            request: The Ollama GenerateRequest to convert
            images: List of image URLs to include

        Returns:
            VisionRequest: The converted OpenAI vision request
        """
        # Extract options from Ollama request
        options = request.options or {}

        # Map Ollama parameters to OpenAI parameters
        temperature = options.get("temperature")
        top_p = options.get("top_p")
        max_tokens = options.get("num_predict")

        # Create a user message with text and images
        content_items = [TextContent(type="text", text=request.prompt)]

        # Add image content items
        for image_url in images:
            content_items.append(
                ImageContent(type="image", image_url=image_url)
            )

        # Create messages array with system and user messages
        messages = []

        # Add system message if present
        if request.system:
            messages.append(
                Message(role="system", content=request.system)
            )

        # Add user message with content items
        messages.append(
            Message(role="user", content=content_items)
        )

        # Create OpenAI vision request
        return VisionRequest(
            model=request.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=request.stream
        )

    @staticmethod
    def convert_chat_with_images_to_vision(
        request: OllamaChatRequest, 
        images: List[str]
    ) -> VisionRequest:
        """
        Convert an Ollama ChatRequest with images to an OpenAI VisionRequest.

        Args:
            request: The Ollama ChatRequest to convert
            images: List of image URLs to include

        Returns:
            VisionRequest: The converted OpenAI vision request
        """
        # Extract options from Ollama request
        options = request.options or {}

        # Map Ollama parameters to OpenAI parameters
        temperature = options.get("temperature")
        top_p = options.get("top_p")
        max_tokens = options.get("num_predict")

        # Process messages
        messages = list(request.messages)

        # For the last user message, convert to multimodal format if it's not already
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                if isinstance(messages[i].content, str):
                    # Convert to multimodal content
                    content_items = [TextContent(type="text", text=messages[i].content)]

                    # Add image content
                    for image_url in images:
                        content_items.append(
                            ImageContent(type="image", image_url=image_url)
                        )

                    # Replace the message with a new one with multimodal content
                    messages[i] = Message(
                        role=messages[i].role,
                        content=content_items,
                        name=messages[i].name
                    )
                break

        # Create OpenAI vision request
        return VisionRequest(
            model=request.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=request.stream
        )