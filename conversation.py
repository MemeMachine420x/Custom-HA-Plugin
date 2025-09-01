"""The conversation platform for the Ollama integration."""

from __future__ import annotations

import logging
from typing import Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import OllamaConfigEntry
from .const import CONF_PROMPT, DOMAIN
from .entity import OllamaBaseLLMEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OllamaConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue

        async_add_entities(
            [OllamaConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class OllamaConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    OllamaBaseLLMEntity,
):
    """Ollama conversation agent."""

    _attr_supports_streaming = True

    def __init__(self, entry: OllamaConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API."""
        settings = {**self.entry.data, **self.subentry.data}

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                settings.get(CONF_LLM_HASS_API),
                settings.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        # Debug: Check what's in the chat log
        _LOGGER.debug("Chat log content after processing: %s", [type(c).__name__ for c in chat_log.content])
        
        # Find the last assistant content
        response_text = ""
        for content in reversed(chat_log.content):
            if isinstance(content, conversation.AssistantContent):
                response_text = content.content or ""
                break
        
        # Create a mock response object with as_dict method
        class MockResponse:
            def __init__(self, content: str):
                self.content = content
            
            def as_dict(self) -> dict:
                return {"content": self.content}
        
        # Return a simple conversation result since we've already processed the message
        return conversation.ConversationResult(
            response=MockResponse(response_text),
            conversation_id=chat_log.conversation_id,
        )
