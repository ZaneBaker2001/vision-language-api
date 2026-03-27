from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vision Language API"
    app_version: str = "1.1.0"

    caption_model_name: str = Field(
        default="Salesforce/blip-image-captioning-base"
    )
    vqa_model_name: str = Field(
        default="Salesforce/blip-vqa-base"
    )

    max_caption_tokens: int = 40
    max_answer_tokens: int = 20
    caption_num_beams: int = 4

    max_upload_size_mb: int = 10
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()