"""Models for Custom LLM integration."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MessageRole(StrEnum):
    """Role of a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class MessageHistory:
    """Chat message history."""

    messages: list[dict[str, Any]]
    """List of message history, including system prompt and assistant responses."""

    @property
    def num_user_messages(self) -> int:
        """Return a count of user messages."""
        return sum(m.get("role") == MessageRole.USER.value for m in self.messages)
