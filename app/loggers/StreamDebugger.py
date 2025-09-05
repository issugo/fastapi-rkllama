import logging

from loggers import DEBUG_MODE

logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logger = logging.getLogger("rkllama.debug_utils")


class StreamDebugger:
    """Utility class for debugging streaming responses"""

    def __init__(self, stream_name="unnamed"):
        self.stream_name = stream_name
        self.chunks = []

    def add_chunk(self, chunk):
        """Add a chunk to the debug log"""
        self.chunks.append(chunk)
        if DEBUG_MODE:
            logger.debug(
                f"Stream '{self.stream_name}' chunk {len(self.chunks)}: {chunk[:50]}..."
            )

    def get_summary(self):
        """Get a summary of the stream"""
        return {
            "stream_name": self.stream_name,
            "chunks": len(self.chunks),
            "total_length": sum(len(c) for c in self.chunks),
            "last_chunk": self.chunks[-1] if self.chunks else None,
        }
