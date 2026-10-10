from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_model: str = "Qwen3.8-Flash-Next"

settings = Settings()