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
from .models import MessageHistory, MessageRole

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

_LOGGER = logging.getLogger(__name__)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"type": "function", "function": tool_spec}


def _fix_invalid_arguments(value: Any) -> Any:
    """Attempt to repair incorrectly formatted json function arguments.

    Small models (for example llama3.1 8B) may produce invalid argument values
    which we attempt to repair here.
    """
    if not isinstance(value, str):
        return value
    if (value.startswith("[") and value.endswith("]")) or (
        value.startswith("{") and value.endswith("}")
    ):
        try:
            return json.loads(value)
        except json.decoder.JSONDecodeError:
            pass
    return value


def _parse_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ollama tool arguments.

    This function improves tool use quality by fixing common mistakes made by
    small local tool use models. This will repair invalid json arguments and
    omit unnecessary arguments with empty values that will fail intent parsing.
    """
    return {k: _fix_invalid_arguments(v) for k, v in arguments.items() if v}


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
    if isinstance(chat_content, conversation.ToolResultContent):
        # Safely handle tool_result that might be None
        tool_result = getattr(chat_content, "tool_result", None) or {}
        return ollama.Message(
            role=MessageRole.TOOL.value,
            content=json.dumps(tool_result),
        )
    if isinstance(chat_content, conversation.AssistantContent):
        # Safely handle tool_calls that might be None
        tool_calls = getattr(chat_content, "tool_calls", None)
        ollama_tool_calls = []
        if tool_calls is not None:
            ollama_tool_calls = [
                ollama.Message.ToolCall(
                    function=ollama.Message.ToolCall.Function(
                        name=tool_call.tool_name,
                        arguments=tool_call.tool_args,
                    )
                )
                for tool_call in tool_calls
            ]
        # Safely handle content that might be None
        content = getattr(chat_content, "content", "") or ""
        return ollama.Message(
            role=MessageRole.ASSISTANT.value,
            content=content,
            tool_calls=ollama_tool_calls or None,
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
        if (tool_calls := response_message.get("tool_calls")) is not None:
            chunk["tool_calls"] = [
                llm.ToolInput(
                    tool_name=tool_call["function"]["name"],
                    tool_args=_parse_tool_args(tool_call["function"]["arguments"]),
                )
                for tool_call in tool_calls
            ]
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
    ) -> None:
        """Generate an answer for the chat log."""
        settings = {**self.entry.data, **self.subentry.data}

        client = self.entry.runtime_data
        model = settings[CONF_MODEL]

        tools: list[dict[str, Any]] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Filter out None content and log any issues
        valid_content = []
        for content in chat_log.content:
            if content is None:
                _LOGGER.warning("Found None content in chat_log.content, skipping")
                continue
            valid_content.append(content)
        
        message_history: MessageHistory = MessageHistory(
            [_convert_content(content) for content in valid_content]
        )
        max_messages = int(settings.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
        self._trim_history(message_history, max_messages)

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

        # Get response
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response_generator = await client.chat(
                    model=model,
                    # Make a copy of the messages because we mutate the list later
                    messages=list(message_history.messages),
                    tools=tools,
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

            # Process streaming content and filter out None values
            streaming_content = []
            async for content in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_stream(response_generator)
            ):
                if content is None:
                    _LOGGER.warning("Found None content in streaming response, skipping")
                    continue
                streaming_content.append(content)
            
            message_history.messages.extend(
                [_convert_content(content) for content in streaming_content]
            )

            if not chat_log.unresponded_tool_results:
                break

    def _trim_history(self, message_history: MessageHistory, max_messages: int) -> None:
        """Trims excess messages from a single history.

        This sets the max history to allow a configurable size history may take
        up in the context window.

        Note that some messages in the history may not be from ollama only, and
        may come from other anents, so the assumptions here may not strictly hold,
        but generally should be effective.
        """
        if max_messages < 1:
            # Keep all messages
            return

        # Ignore the in progress user message
        num_previous_rounds = message_history.num_user_messages - 1
        if num_previous_rounds >= max_messages:
            # Trim history but keep system prompt (first message).
            # Every other message should be an assistant message, so keep 2x
            # message objects. Also keep the last in progress user message
            num_keep = 2 * max_messages + 1
            drop_index = len(message_history.messages) - num_keep
            message_history.messages = [
                message_history.messages[0],
                *message_history.messages[drop_index:],
            ]
