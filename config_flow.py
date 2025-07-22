"""Config flow for Custom LLM integration."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any
from collections.abc import Mapping

import httpx
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
    ConfigEntryState,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_NAME, CONF_URL
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.util.ssl import get_default_context

from .const import (
    CONF_KEEP_ALIVE,
    CONF_MAX_HISTORY,
    CONF_MODEL,
    CONF_NUM_CTX,
    CONF_PROMPT,
    CONF_THINK,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_THINK,
    DEFAULT_TIMEOUT,
    DOMAIN,
    MAX_NUM_CTX,
    MIN_NUM_CTX,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
    }
)


class CustomLLMConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Custom LLM."""

    VERSION = 3
    MINOR_VERSION = 2

    def __init__(self) -> None:
        self.url: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}
        url = user_input[CONF_URL]

        self._async_abort_entries_match({CONF_URL: url})

        try:
            url = cv.url(url)
        except vol.Invalid:
            errors["base"] = "invalid_url"
            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        try:
            client = httpx.AsyncClient(base_url=url, verify=get_default_context())
            async with asyncio.timeout(DEFAULT_TIMEOUT):
                response = await client.post("/generate", json={"prompt": "ping"})
                response.raise_for_status()
        except (httpx.HTTPError, asyncio.TimeoutError):
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"

        if errors:
            return self.async_show_form(
                step_id="user",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA, user_input
                ),
                errors=errors,
            )

        return self.async_create_entry(title=url, data={CONF_URL: url})

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            "conversation": CustomLLMSubentryFlowHandler,
            "ai_task_data": CustomLLMSubentryFlowHandler,
        }


class CustomLLMSubentryFlowHandler(ConfigSubentryFlow):
    """Flow for managing Custom LLM subentries."""

    def __init__(self) -> None:
        super().__init__()
        self._name: str | None = None
        self._model: str | None = None

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_set_options(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle model selection and configuration step."""
        if self._get_entry().state != ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is None:
            models_to_list = [
                SelectOptionDict(label="custom-llm", value="custom-llm"),
                SelectOptionDict(label="custom-llm-v2", value="custom-llm-v2"),
            ]

            options = {} if self._is_new else self._get_reconfigure_subentry().data.copy()

            return self.async_show