import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_CURRENT_DIR)
_BACKEND_DIR = os.path.dirname(_APP_DIR)
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)

# Load .env from root directory or backend directory
load_dotenv(os.path.join(_ROOT_DIR, ".env"))
load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
load_dotenv(".env")


class Settings(BaseSettings):
    PROJECT_NAME: str = "DocRAG"

    # Vector DB
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"

    # Embedding model (runs locally, free)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Google Gemini API Key and Model (RAG Answer Generation)
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"

    # Groq API Key (Ragas Judge Evaluation)
    GROQ_API_KEY: str = ""

    class Config:
        env_file = (
            os.path.join(_ROOT_DIR, ".env"),
            os.path.join(_BACKEND_DIR, ".env"),
            ".env",
        )
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

_RESOLVED_GEMINI_MODEL: str | None = None


def get_active_gemini_model(api_key: str | None = None) -> str:
    """
    Dynamically lists models supported by the Google API key, filters out experimental
    low-quota models, and verifies candidate models with a lightweight probe.
    """
    global _RESOLVED_GEMINI_MODEL
    if _RESOLVED_GEMINI_MODEL:
        return _RESOLVED_GEMINI_MODEL

    key = api_key or settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return settings.GEMINI_MODEL_NAME

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        available_models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        print(f"[GeminiModelResolver] Available models on API key: {available_models}")

        preferred_order = [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            "gemini-3.6-flash",
            "gemini-3.7-flash",
        ]

        candidates = [m for m in preferred_order if m in available_models]
        for m in available_models:
            if m not in candidates:
                candidates.append(m)
        if not candidates and available_models:
            candidates = available_models

        # Probe candidate models to guarantee existence and available quota
        for candidate in candidates:
            try:
                test_model = genai.GenerativeModel(candidate)
                resp = test_model.generate_content("ping")
                if resp and resp.text:
                    _RESOLVED_GEMINI_MODEL = candidate
                    print(f"[GeminiModelResolver] Successfully verified active Gemini model: '{candidate}'")
                    return candidate
            except Exception as probe_err:
                print(f"[GeminiModelResolver] Candidate '{candidate}' skipped: {probe_err}")

    except Exception as e:
        print(f"[GeminiModelResolver] Warning: Could not list models ({e}).")

    _RESOLVED_GEMINI_MODEL = "gemini-2.5-flash-lite"
    print(f"[GeminiModelResolver] Fallback model selected: '{_RESOLVED_GEMINI_MODEL}'")
    return _RESOLVED_GEMINI_MODEL

_RESOLVED_GROQ_MODEL: str | None = None


def get_active_groq_model(api_key: str | None = None) -> str:
    """
    Dynamically fetches supported Groq models via API and probes candidate models
    to guarantee zero 404 / decommissioned errors.
    """
    global _RESOLVED_GROQ_MODEL
    if _RESOLVED_GROQ_MODEL:
        return _RESOLVED_GROQ_MODEL

    key = api_key or settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
    if not key:
        return "openai/gpt-oss-120b"

    # Attempt to dynamically list active models from Groq API endpoint
    available_models = []
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            available_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            print(f"[GroqModelResolver] Active models on Groq key: {available_models}")
    except Exception as api_err:
        print(f"[GroqModelResolver] Warning: Could not list Groq models ({api_err}).")

    preferred_candidates = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
    ]

    candidates = [m for m in preferred_candidates if m in available_models]
    for m in available_models:
        if m not in candidates and "whisper" not in m and "vision" not in m:
            candidates.append(m)

    if not candidates:
        candidates = preferred_candidates

    from langchain_groq import ChatGroq
    for candidate in candidates:
        try:
            chat = ChatGroq(
                model=candidate,
                groq_api_key=key,
                temperature=0.0,
                max_retries=1,
                request_timeout=15,
            )
            res = chat.invoke("ping")
            if res and res.content:
                _RESOLVED_GROQ_MODEL = candidate
                print(f"[GroqModelResolver] Successfully verified active Groq model: '{candidate}'")
                return candidate
        except Exception as probe_err:
            print(f"[GroqModelResolver] Candidate '{candidate}' skipped: {probe_err}")

    _RESOLVED_GROQ_MODEL = candidates[0] if candidates else "openai/gpt-oss-20b"
    return _RESOLVED_GROQ_MODEL


