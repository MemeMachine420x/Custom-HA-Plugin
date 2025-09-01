"""Base entity for the Ollama integration."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
import json
import logging
from typing import Any

import ollama
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity

from . import OllamaConfigEntry
from .const import (
    CONF_KEEP_ALIVE,
    CONF_MAX_HISTORY,
    CONF_MODEL,
    CONF_NUM_CTX,
    CONF_THINK,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_NUM_CTX,
    DOMAIN,
)
from .models import MessageRole

# Tool iterations removed since proxy doesn't handle tool calls

_LOGGER = logging.getLogger(__name__)








def _convert_content(
    chat_content: (
        conversation.Content
        | conversation.ToolResultContent
        | conversation.AssistantContent
        | None
    ),
) -> ollama.Message:
    """Create tool response content."""
    if chat_content is None:
        # Handle None content gracefully
        return ollama.Message(
            role=MessageRole.ASSISTANT.value,
            content="",
        )

    if isinstance(chat_content, conversation.AssistantContent):
        # Safely handle content that might be None
        content = getattr(chat_content, "content", "") or ""
        return ollama.Message(
            role=MessageRole.ASSISTANT.value,
            content=content,
        )
    if isinstance(chat_content, conversation.UserContent):
        images: list[ollama.Image] = []
        # Safely handle attachments that might be None
        attachments = getattr(chat_content, "attachments", None)
        if attachments is not None:
            for attachment in attachments:
                if not attachment.mime_type.startswith("image/"):
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="unsupported_attachment_type",
                    )
                images.append(ollama.Image(value=attachment.path))
        # Safely handle content that might be None
        content = getattr(chat_content, "content", "") or ""
        return ollama.Message(
            role=MessageRole.USER.value,
            content=content,
            images=images or None,
        )
    if isinstance(chat_content, conversation.SystemContent):
        # Safely handle content that might be None
        content = getattr(chat_content, "content", "") or ""
        return ollama.Message(
            role=MessageRole.SYSTEM.value,
            content=content,
        )
    raise TypeError(f"Unexpected content type: {type(chat_content)}")


async def _transform_stream(
    result: AsyncIterator[ollama.ChatResponse],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the response stream into HA format.

    An Ollama streaming response may come in chunks like this:

    response: message=Message(role="assistant", content="Paris")
    response: message=Message(role="assistant", content=".")
    response: message=Message(role="assistant", content=""), done: True, done_reason: "stop"
    response: message=Message(role="assistant", tool_calls=[...])
    response: message=Message(role="assistant", content=""), done: True, done_reason: "stop"

    This generator conforms to the chatlog delta stream expectations in that it
    yields deltas, then the role only once the response is done.
    """

    new_msg = True
    async for response in result:
        _LOGGER.debug("Received response: %s", response)
        response_message = response["message"]
        chunk: conversation.AssistantContentDeltaDict = {}
        if new_msg:
            new_msg = False
            chunk["role"] = "assistant"

        if (content := response_message.get("content")) is not None:
            chunk["content"] = content
        if response_message.get("done"):
            new_msg = True
        yield chunk


class OllamaBaseLLMEntity(Entity):
    """Ollama base LLM entity."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id

        model, _, version = subentry.data[CONF_MODEL].partition(":")
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Ollama",
            model=model,
            sw_version=version or "latest",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: vol.Schema | None = None,
    ) -> conversation.ConversationResult:
        """Generate an answer for the chat log."""
        _LOGGER.info("Starting _async_handle_chat_log - chat_log.content length: %d", len(chat_log.content))
        
        # Prevent infinite loops by checking if we're already processing
        if hasattr(self, '_processing_chat_log') and self._processing_chat_log:
            _LOGGER.warning("Already processing chat log, skipping to prevent infinite loop")
            return conversation.ConversationResult(
                response="",
                conversation_id=chat_log.conversation_id,
            )
        
        # Set processing flag to prevent infinite loops
        self._processing_chat_log = True
        
        try:
            settings = {**self.entry.data, **self.subentry.data}
            client = self.entry.runtime_data
            model = settings[CONF_MODEL]

            # Disable tools since proxy doesn't handle them
            tools = None

            # Convert chat log content to Ollama message format
            messages = []
            for content in chat_log.content:
                if content is None:
                    _LOGGER.warning("Found None content in chat_log.content, skipping")
                    continue
                messages.append(_convert_content(content))
            
            # Apply history trimming
            max_messages = int(settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
            if max_messages > 0 and len(messages) > max_messages * 2 + 1:
                # Keep system prompt (first message) and recent messages
                messages = [messages[0]] + messages[-(max_messages * 2):]
            
            _LOGGER.info("Sending %d messages to Ollama", len(messages))
            for i, msg in enumerate(messages):
                _LOGGER.debug("Message %d: role=%s, content_length=%d", i, msg.get("role", "unknown"), len(str(msg.get("content", ""))))

            output_format: dict[str, Any] | None = None
            if structure:
                output_format = convert(
                    structure,
                    custom_serializer=(
                        chat_log.llm_api.custom_serializer
                        if chat_log.llm_api
                        else llm.selector_serializer
                    ),
                )

            # Get response - single iteration since proxy doesn't handle tool calls
            try:
                response_generator = await client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    # keep_alive requires specifying unit. In this case, seconds
                    keep_alive=f"{settings.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)}s",
                    options={CONF_NUM_CTX: settings.get(CONF_NUM_CTX, DEFAULT_NUM_CTX)},
                    think=settings.get(CONF_THINK),
                    format=output_format,
                )
            except (ollama.RequestError, ollama.ResponseError) as err:
                _LOGGER.error("Unexpected error talking to Ollama server: %s", err)
                raise HomeAssistantError(
                    f"Sorry, I had a problem talking to the Ollama server: {err}"
                ) from err

            # Process streaming content - Home Assistant handles adding to chat log
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_stream(response_generator)
            ):
                if content is None:
                    _LOGGER.warning("Found None content in streaming response, skipping")
                    continue
            
            _LOGGER.info("Finished _async_handle_chat_log - chat_log.content length: %d", len(chat_log.content))
            
            # Return a conversation result with the response
            # Get the last assistant content from the chat log
            response_text = ""
            for content in reversed(chat_log.content):
                if isinstance(content, conversation.AssistantContent) and content.content:
                    response_text = content.content
                    break
            
            return conversation.ConversationResult(
                response=response_text,
                conversation_id=chat_log.conversation_id,
            )
        
        finally:
            self._processing_chat_log = False