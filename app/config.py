from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    chunk_size: int = 800
    chunk_overlap: int = 100


settings = Settings()
