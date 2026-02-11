from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "DocRAG"

    # Vector DB
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"

    # Embedding model (runs locally, free)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Google Gemini API Key
    GOOGLE_API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
