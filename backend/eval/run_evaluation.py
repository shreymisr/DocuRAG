"""
eval/run_evaluation.py — End-to-End Benchmark: Baseline vs Upgraded RAG
========================================================================

Compares two RAG retrieval & generation pipelines across the benchmark dataset:
  1. BASELINE: Dense-only retrieval (vector_store.similarity_search, k=4) -> Gemini generation
  2. UPGRADED: Hybrid search (k=10) -> Cross-Encoder reranking (top_k=4) -> Gemini generation

Evaluates both pipelines using Ragas metrics with Google Gemini as Judge:
  - Faithfulness (Factual consistency with context / Hallucination detection)
  - Answer Relevancy (Directness & relevance to the prompt)
  - Context Precision (Signal-to-noise / ranking quality of retrieved chunks)
  - Context Recall (Coverage of ground-truth information)

Outputs:
  - eval/results/baseline.json
  - eval/results/upgraded.json
  - Console summary table comparing averages and percentage improvements.

Run from backend/ directory:
    python -m eval.run_evaluation
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Path hygiene — ensure backend/ is on sys.path
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from app.services.rag_chain import PROMPT_TEMPLATE, RAGService
from app.services.reranker import RerankerService
from app.services.vector_store import VectorStoreService
from eval.ragas_setup import get_ragas_embeddings, get_ragas_llm

DATASET_PATH = os.path.join(_BACKEND_DIR, "eval", "qa_dataset.json")
RESULTS_DIR = os.path.join(_BACKEND_DIR, "eval", "results")
BASELINE_OUTPUT = os.path.join(RESULTS_DIR, "baseline.json")
UPGRADED_OUTPUT = os.path.join(RESULTS_DIR, "upgraded.json")

# Default delay (in seconds) between sequential Gemini API calls to respect free-tier limits
CALL_INTERVAL_SECONDS = 4.2


def parse_args():
    parser = argparse.ArgumentParser(description="Run DocRAG Ragas Benchmark Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Limit evaluation to the first N questions (default: 6 for full completion within free-tier token limits)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=CALL_INTERVAL_SECONDS,
        help=f"Delay in seconds between API calls to avoid rate limits (default: {CALL_INTERVAL_SECONDS}s)",
    )
    return parser.parse_args()


def load_benchmark_dataset(path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Loads the benchmark QA dataset from JSON, optionally truncated to `limit`."""
    if not os.path.exists(path):
        draft_path = os.path.join(_BACKEND_DIR, "eval", "qa_dataset_draft.json")
        if os.path.exists(draft_path):
            print(f"[Warning] '{path}' not found. Falling back to '{draft_path}'.")
            path = draft_path
        else:
            raise FileNotFoundError(f"QA dataset not found at '{path}' or '{draft_path}'.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit and limit > 0:
        data = data[:limit]
        print(f"✓ Loaded first {len(data)} test question(s) from '{path}' (--limit {limit}).")
    else:
        print(f"✓ Loaded {len(data)} test question(s) from '{path}'.")
    return data


def generate_with_retry(model, prompt: str, max_retries: int = 5, initial_delay: float = 4.0) -> str:
    """Generates non-streaming content via Gemini with exponential backoff on rate limits."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resourceexhausted" in err_str or "quota" in err_str:
                if "per day" in err_str or "per_day" in err_str or "daily" in err_str:
                    print("\n[Gemini Quota Exceeded] Daily quota (1500 RPD) has been exhausted for this Google API Key.")
                    print("Please update GOOGLE_API_KEY in .env with a fresh key.")
                    return f"Error: Daily quota exhausted ({e})"
                print(f"  [Rate limit 429] Pacing backoff: waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 1.5
            else:
                print(f"  [Generation error]: {e}")
                if attempt == max_retries - 1:
                    return f"Error: {e}"
                time.sleep(delay)
    return ""


def run_pipeline_baseline(
    qa_pairs: List[Dict[str, Any]],
    vs: VectorStoreService,
    rag_service: RAGService,
    delay: float = CALL_INTERVAL_SECONDS,
) -> Dict[str, List[Any]]:
    """Runs BASELINE: Dense-only retrieval (vector_store.similarity_search, k=4) -> Gemini generation."""
    questions, answers, contexts_list, ground_truths = [], [], [], []

    print("\n" + "═" * 70)
    print("▶ Running Pipeline 1/2: BASELINE (Dense-only retrieval, k=4)")
    print("═" * 70)

    for idx, item in enumerate(qa_pairs, start=1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"[{idx}/{len(qa_pairs)}] Question: {q[:60]}...")

        # 1. Dense-only retrieval (vector_store.similarity_search, k=4)
        dense_docs = vs.similarity_search(q, k=4)
        doc_texts = [d.page_content for d in dense_docs] if dense_docs else [""]

        # 2. Context formatting
        context_str = "\n\n".join(doc_texts)

        # 3. LLM Answer Generation (non-streaming Gemini call)
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=q)
        answer = generate_with_retry(rag_service.model, prompt)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(doc_texts)
        ground_truths.append(gt)

        # Pacing between requests
        time.sleep(delay)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def run_pipeline_upgraded(
    qa_pairs: List[Dict[str, Any]],
    vs: VectorStoreService,
    reranker: RerankerService,
    rag_service: RAGService,
    delay: float = CALL_INTERVAL_SECONDS,
) -> Dict[str, List[Any]]:
    """Runs UPGRADED: hybrid_search(k=10) -> RerankerService.rerank(top_k=4) -> Gemini generation."""
    questions, answers, contexts_list, ground_truths = [], [], [], []

    print("\n" + "═" * 70)
    print("▶ Running Pipeline 2/2: UPGRADED (Hybrid k=10 -> Reranker top_k=4)")
    print("═" * 70)

    for idx, item in enumerate(qa_pairs, start=1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"[{idx}/{len(qa_pairs)}] Question: {q[:60]}...")

        # 1. Stage-1 Hybrid Search (k=10)
        candidates = vs.hybrid_search(q, k=10)

        # 2. Stage-2 Cross-Encoder Reranking (top_k=4)
        if candidates:
            reranked_docs = reranker.rerank(q, candidates, top_k=4)
            doc_texts = [d.page_content for d in reranked_docs]
        else:
            doc_texts = [""]

        # 3. Context formatting
        context_str = "\n\n".join(doc_texts)

        # 4. LLM Answer Generation (non-streaming Gemini call)
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=q)
        answer = generate_with_retry(rag_service.model, prompt)

        questions.append(q)
        answers.append(answer)
        contexts_list.append(doc_texts)
        ground_truths.append(gt)

        # Pacing between requests
        time.sleep(delay)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def evaluate_with_ragas(
    data_dict: Dict[str, List[Any]], pipeline_name: str, call_interval: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates dataset using Ragas metrics (faithfulness, answer_relevancy, context_precision, context_recall)
    and the Groq judge (openai/gpt-oss-120b) from ragas_setup.py.

    With Groq's high rate limits (500 RPM for gpt-oss-120b), RunConfig max_workers is set to 2.
    Includes delay and retry logic as a safety net for transient rate limit 429s.
    """
    print(f"\n[Ragas Evaluation] Evaluating {pipeline_name} with Groq Judge (openai/gpt-oss-120b)...")

    judge_llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    run_config = RunConfig(max_workers=2, timeout=120, max_retries=3, max_wait=30)
    metrics_config = [
        ("faithfulness", faithfulness),
        ("answer_relevancy", answer_relevancy),
        ("context_precision", context_precision),
        ("context_recall", context_recall),
    ]

    # Groq API strictly enforces 'n' <= 1 for chat completions.
    # Ragas's answer_relevancy metric calls generate_multiple() which passes n > 1 by default.
    # We configure answer_relevancy to generate 1 question or override n in its prompt generator.
    if hasattr(answer_relevancy, "strictness"):
        answer_relevancy.strictness = 1

    n = len(data_dict["question"])
    # Estimate runtime: 4 metrics per question, avg 2 LLM calls per metric * n questions = ~8*n calls at 150 RPM
    est_total_calls = n * 8
    est_seconds = (est_total_calls / 150.0) * 60.0
    est_minutes = est_seconds / 60.0
    print(f"  • Dataset size: {n} questions")
    print(f"  • Estimated Groq LLM calls: ~{est_total_calls} calls across 4 metrics")
    print(f"  • Estimated evaluation time: ~{est_minutes:.1f} minute(s) ({est_seconds:.0f}s at 150 RPM throughput)")

    for _name, m in metrics_config:
        if hasattr(m, "llm"):
            m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings
        m.init(run_config)

    n = len(data_dict["question"])
    per_question_results = []
    total_calls = n * len(metrics_config)
    call_num = 0

    for i in range(n):
        q = data_dict["question"][i]
        ans = data_dict["answer"][i]
        ctxs = data_dict["contexts"][i]
        gt = data_dict["ground_truth"][i]

        row_scores: Dict[str, Any] = {
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            "ground_truth": gt,
        }

        sample = SingleTurnSample(
            user_input=q,
            response=ans,
            retrieved_contexts=ctxs,
            reference=gt,
        )

        for metric_name, metric in metrics_config:
            call_num += 1
            print(f"  [{call_num}/{total_calls}] Scoring '{metric_name}' for Q{i+1}: {q[:50]}...")

            max_retries = 5
            score = float("nan")
            for attempt in range(max_retries):
                try:
                    score = metric.single_turn_score(sample)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "resource_exhausted" in err_str or "rate_limit" in err_str:
                        if "per day" in err_str or "per_day" in err_str or "daily" in err_str:
                            print("  [FATAL] Daily quota exhausted. Update GROQ_API_KEY in .env.")
                            row_scores[metric_name] = float("nan")
                            break
                        wait = 15.0 * (attempt + 1)
                        print(f"  [429 Rate Limit] Waiting {wait:.0f}s for quota window to reset (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        print(f"  [Warning] {metric_name} failed for sample: {e}")
                        score = float("nan")
                        break

            row_scores[metric_name] = score
            if call_num < total_calls:
                time.sleep(call_interval)

        per_question_results.append(row_scores)

    agg: Dict[str, float] = {}
    for metric_name, _ in metrics_config:
        vals = [
            r[metric_name] for r in per_question_results
            if isinstance(r.get(metric_name), float) and not math.isnan(r[metric_name])
        ]
        agg[metric_name] = (sum(vals) / len(vals)) if vals else float("nan")

    return {
        "pipeline": pipeline_name,
        "aggregate_scores": agg,
        "per_question": per_question_results,
    }


def save_json_results(data: Dict[str, Any], filepath: str) -> None:
    """Saves evaluation results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved results to '{filepath}'.")


def print_comparison_table(baseline_scores: Dict[str, float], upgraded_scores: Dict[str, float]) -> None:
    """Prints a formatted comparison table with percentage deltas and clear metric change text."""
    metric_names = [
        ("Faithfulness", "faithfulness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall", "context_recall"),
    ]

    print("\n" + "═" * 78)
    print("               RAG BENCHMARK EVALUATION RESULTS (RAGAS)")
    print("═" * 78)
    print(f" {'Metric':<22} │ {'Baseline (Dense)':<18} │ {'Upgraded (Hybrid+Rerank)':<24} │ {'Delta':<10}")
    print("─" * 78)

    summary_lines = []

    for label, key in metric_names:
        b_val = baseline_scores.get(key, float("nan"))
        u_val = upgraded_scores.get(key, float("nan"))

        b_str = f"{b_val:>16.4f}" if not (isinstance(b_val, float) and math.isnan(b_val)) else f"{'nan':>16}"
        u_str = f"{u_val:>22.4f}" if not (isinstance(u_val, float) and math.isnan(u_val)) else f"{'nan':>22}"

        if not (math.isnan(b_val) or math.isnan(u_val)):
            if b_val > 0:
                delta_pct = ((u_val - b_val) / b_val) * 100
                delta_str = f"{delta_pct:+.1f}%"
            else:
                delta_str = "0.0%" if u_val == 0 else "+100.0%"
            summary_lines.append(f"• {label}: {b_val:.4f} -> {u_val:.4f} ({delta_str})")
        else:
            delta_str = "N/A"
            summary_lines.append(f"• {label}: N/A")

        print(f" {label:<22} │ {b_str} │ {u_str} │ {delta_str:>8}")

    print("═" * 78)
    print("\nSUMMARY OF METRIC IMPROVEMENTS:")
    for line in summary_lines:
        print(f"  {line}")
    print("═" * 78 + "\n")


def main():
    args = parse_args()

    print("\n=======================================================")
    print("       DocRAG End-to-End Evaluation Benchmark")
    print("=======================================================")

    # 1. Load dataset (optionally limited)
    qa_pairs = load_benchmark_dataset(DATASET_PATH, limit=args.limit)

    # 2. Initialize services
    print("\nInitializing services (VectorStore, Reranker, RAGService)...")
    vs = VectorStoreService()
    reranker = RerankerService()
    rag_service = RAGService(vector_store_service=vs, reranker_service=reranker)

    # 3. Run Pipeline 1: Baseline (Dense-only retrieval + Gemini generation)
    baseline_data = run_pipeline_baseline(qa_pairs, vs, rag_service, delay=args.delay)

    # 4. Run Pipeline 2: Upgraded (Hybrid search + Reranking + Gemini generation)
    upgraded_data = run_pipeline_upgraded(qa_pairs, vs, reranker, rag_service, delay=args.delay)

    # 5. Evaluate Baseline with Ragas
    baseline_eval = evaluate_with_ragas(baseline_data, "Baseline (Dense-only k=4)", call_interval=args.delay)
    save_json_results(baseline_eval, BASELINE_OUTPUT)

    # 6. Evaluate Upgraded with Ragas
    upgraded_eval = evaluate_with_ragas(upgraded_data, "Upgraded (Hybrid k=10 + Rerank top_k=4)", call_interval=args.delay)
    save_json_results(upgraded_eval, UPGRADED_OUTPUT)

    # 7. Print Final Comparison Table & Summary
    print_comparison_table(
        baseline_eval["aggregate_scores"],
        upgraded_eval["aggregate_scores"],
    )


if __name__ == "__main__":
    main()
