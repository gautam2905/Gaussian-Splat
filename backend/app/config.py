from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

OrchestratorMode = Literal["stub", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # OpenAI (per Amendment K)
    openai_api_key: str = ""
    orchestrator_model: str = "gpt-4o-mini"
    orchestrator_mode: OrchestratorMode = "stub"  # default to stub for $0 dev

    # Anthropic (reserved, unused in v1)
    anthropic_api_key: str = ""

    modal_app_name: str = "museums-ml"
    data_dir: str = "/data"

    # Object generation mode (per Amendment K + budget pivot)
    object_gen_mode: Literal["stub", "modal"] = "stub"

    max_objects_per_exhibit: int = 5
    min_objects_per_exhibit: int = 3
    per_ip_daily_cap: int = 5
    daily_modal_budget_usd: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
