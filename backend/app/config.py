from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    modal_app_name: str = "museums-ml"
    data_dir: str = "/data"
    max_objects_per_exhibit: int = 5
    min_objects_per_exhibit: int = 3
    per_ip_daily_cap: int = 5
    daily_modal_budget_usd: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
