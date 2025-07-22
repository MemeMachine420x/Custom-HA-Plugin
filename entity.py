"""Base entity for Custom LLM integration."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity import Entity

from . import CustomLLMConfigEntry
from .const import CONF_MODEL, DOMAIN

_LOGGER = logging.getLogger(__name__)


class BaseLLMEntity(Entity):
    """Base class for Custom LLM AI entities."""

    def __init__(self, entry: CustomLLMConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_name = subentry.title
        self._attr_unique_id = subentry.subentry_id

        model = subentry.data.get(CONF_MODEL, "custom-llm")

        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="CustomLLM",
            model=model,
            sw_version="1.0",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure: Any = None,
    ) -> None:
        """Generate a response for the chat log using a custom LLM."""
        client: httpx.AsyncClient = self.entry.runtime_data

        # Use the latest user message as the prompt
        prompt = ""
        for content in reversed(chat_log.content):
            if isinstance(content, conversation.UserContent):
                prompt = content.content
                break

        if not prompt:
            raise HomeAssistantError("No user prompt found in chat log")

        try:
            response = await client.post("/generate", json={"prompt": prompt})
            response.raise_for_status()
        except httpx.HTTPError as err:
            _LOGGER.error("Error communicating with LLM server: %s", err)
            raise HomeAssistantError(f"LLM server error: {err}") from err

        result = response.json()
        chat_log.add_assistant_message(result.get("response", ""))
