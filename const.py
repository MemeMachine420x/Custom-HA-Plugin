"""Constants for the Custom LLM integration."""

DOMAIN = "custom_llm"

DEFAULT_NAME = "Custom LLM"

# Core config keys
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_THINK = "think"
CONF_KEEP_ALIVE = "keep_alive"
CONF_NUM_CTX = "num_ctx"
CONF_MAX_HISTORY = "max_history"

# Default values
DEFAULT_MODEL = "custom-llm"
DEFAULT_PROMPT = ""
DEFAULT_THINK = False
DEFAULT_KEEP_ALIVE = -1  # seconds. -1 = indefinite, 0 = never
DEFAULT_NUM_CTX = 8192
DEFAULT_MAX_HISTORY = 20
DEFAULT_TIMEOUT = 5.0  # seconds

KEEP_ALIVE_FOREVER = -1
MAX_HISTORY_SECONDS = 60 * 60  # 1 hour

MIN_NUM_CTX = 2048
MAX_NUM_CTX = 131072

# Friendly default names
DEFAULT_CONVERSATION_NAME = "Custom LLM Conversation"
DEFAULT_AI_TASK_NAME = "Custom LLM AI Task"

# Recommended configuration
RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_MAX_HISTORY: DEFAULT_MAX_HISTORY,
}
