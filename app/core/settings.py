import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "vkassist"
    max_query_characters: int = 1_000
    default_limit: int = 5
    max_limit: int = 20
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str | None = os.getenv("LLM_MODEL")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "600"))


settings = Settings()
