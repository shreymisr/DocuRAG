"""
eval/ragas_setup.py — Ragas Evaluation Setup with Groq (openai/gpt-oss-120b) Judge
=================================================================================

Configures Ragas to use Groq (openai/gpt-oss-120b) as the evaluator / judge LLM
via LangchainLLMWrapper and ChatGroq, reusing the existing project settings
and Groq API key configuration.

Exposes:
  - get_ragas_llm(): Returns the singleton LangchainLLMWrapper for the Groq judge.
  - get_ragas_embeddings(): Returns the HuggingFace embeddings wrapper for Ragas metrics.

Run directly to execute a smoke test verifying the Groq judge wiring:
    python -m eval.ragas_setup
"""

import os
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Path hygiene — ensure backend/ is on sys.path
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from app.core.config import settings, get_active_groq_model

# Singletons to prevent reinitializing models on every evaluation step
_RAGAS_LLM_INSTANCE: Optional[LangchainLLMWrapper] = None
_RAGAS_EMBEDDINGS_INSTANCE: Optional[LangchainEmbeddingsWrapper] = None


import asyncio
import time

class GroqRateLimiter:
    """
    Strict rate limiter enforcing inter-request spacing and rolling RPM ceiling
    with 429 retry backoff for Groq API calls.
    """
    def __init__(self, rpm_limit: float = 25.0):
        self.interval = 60.0 / rpm_limit
        self.last_call_time = 0.0
        self._async_lock = asyncio.Lock()

    async def acquire_async(self):
        async with self._async_lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                await asyncio.sleep(wait_time)
            self.last_call_time = time.time()

    def acquire_sync(self):
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.interval:
            wait_time = self.interval - elapsed
            time.sleep(wait_time)
        self.last_call_time = time.time()

_GROQ_RATE_LIMITER = GroqRateLimiter(rpm_limit=25.0)


def get_ragas_llm(model_name: Optional[str] = None) -> LangchainLLMWrapper:
    """
    Returns a configured, reusable LangchainLLMWrapper instance using Groq (openai/gpt-oss-120b) as the judge.

    Groq is used as an independent judge model (different model family than the Gemini generator)
    to avoid self-preference bias, and to work within free-tier rate limits
    (Groq gpt-oss-120b: 500 RPM vs Gemini's 15 RPM / 1,000 RPD).
    """
    global _RAGAS_LLM_INSTANCE

    if _RAGAS_LLM_INSTANCE is None:
        api_key = getattr(settings, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API Key is required for Ragas evaluation. "
                "Set GROQ_API_KEY in .env or your environment."
            )

        resolved_model = model_name or get_active_groq_model(api_key=api_key)

        chat_groq = ChatGroq(
            model=resolved_model,
            groq_api_key=api_key,
            temperature=0.0,
            max_retries=3,          # Let SDK retry on transient errors (network blips)
            request_timeout=90,     # Per-request timeout
        )
        wrapper = LangchainLLMWrapper(chat_groq)

        # Groq API strictly enforces `n` <= 1 for chat completions.
        # Patch wrapper's internal generate methods so any `n > 1` request from Ragas is capped at 1,
        # and enforce rate-limiting with 429 exponential backoff.
        original_agenerate_text = wrapper.agenerate_text
        async def patched_agenerate_text(prompt, n=1, temperature=None, stop=None, callbacks=None, **kwargs):
            kwargs.pop("max_tokens", None)
            await _GROQ_RATE_LIMITER.acquire_async()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return await original_agenerate_text(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    if ("429" in err_str or "rate_limit" in err_str or "resource_exhausted" in err_str) and attempt < max_retries - 1:
                        backoff = 10.0 * (2 ** attempt)
                        print(f"  [Groq 429 Rate Limit] Retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(backoff)
                    else:
                        raise e

        wrapper.agenerate_text = patched_agenerate_text

        original_generate_text = wrapper.generate_text
        def patched_generate_text(prompt, n=1, temperature=None, stop=None, callbacks=None, **kwargs):
            kwargs.pop("max_tokens", None)
            _GROQ_RATE_LIMITER.acquire_sync()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return original_generate_text(prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    if ("429" in err_str or "rate_limit" in err_str or "resource_exhausted" in err_str) and attempt < max_retries - 1:
                        backoff = 10.0 * (2 ** attempt)
                        print(f"  [Groq 429 Rate Limit] Retrying in {backoff:.1f}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(backoff)
                    else:
                        raise e

        wrapper.generate_text = patched_generate_text

        _RAGAS_LLM_INSTANCE = wrapper
        print(f"[RagasSetup] Initialized Groq judge LLM ({resolved_model}) with rate limiter & n=1 cap.")

    return _RAGAS_LLM_INSTANCE


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Returns a LangchainEmbeddingsWrapper configured with the project's local
    HuggingFace embedding model (all-MiniLM-L6-v2) for Ragas metrics requiring embeddings.
    """
    global _RAGAS_EMBEDDINGS_INSTANCE

    if _RAGAS_EMBEDDINGS_INSTANCE is None:
        hf_embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        _RAGAS_EMBEDDINGS_INSTANCE = LangchainEmbeddingsWrapper(hf_embeddings)
        print(f"[RagasSetup] Initialized embeddings wrapper ({settings.EMBEDDING_MODEL_NAME}).")

    return _RAGAS_EMBEDDINGS_INSTANCE


if __name__ == "__main__":
    import time
    print("\n--- Running Ragas + Groq Judge (openai/gpt-oss-120b) Smoke Test ---")

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import faithfulness
        from ragas.run_config import RunConfig

        # 1. Initialize
        judge_llm = get_ragas_llm()
        embeddings = get_ragas_embeddings()

        run_config = RunConfig(max_workers=2, timeout=120, max_retries=2, max_wait=30)
        faithfulness.llm = judge_llm
        faithfulness.init(run_config)

        # 2. Hardcoded test sample — use single_turn_score (sync, sequential)
        sample = SingleTurnSample(
            user_input="Where was the first modern Olympic Games held and in what year?",
            retrieved_contexts=[
                "The 1896 Summer Olympics, officially known as the Games of the I Olympiad, "
                "was the first international Olympic Games held in modern history. "
                "It was organized by the International Olympic Committee and took place in Athens, Greece."
            ],
            response="The first modern Olympic Games was held in Athens, Greece in 1896.",
            reference="Athens, Greece in 1896",
        )

        print(f"Scoring faithfulness with Groq judge ({judge_llm.langchain_llm.model_name if hasattr(judge_llm, 'langchain_llm') else 'Groq'})...")
        score = faithfulness.single_turn_score(sample)

        print(f"\n[SMOKE TEST RESULT]")
        print(f"  faithfulness = {score:.4f}")
        print("\n✓ Ragas + Groq judge wiring works successfully!")

    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
