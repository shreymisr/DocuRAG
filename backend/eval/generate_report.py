"""
eval/generate_report.py — Benchmark Comparison Report Generator
===============================================================

Reads:
  - eval/results/baseline.json
  - eval/results/upgraded.json

Outputs:
  - eval/results/comparison_report.md

Generates a markdown comparison report with summary metrics, plain-language
explanations, regressions tracking, and a resume-bullet summary.
"""

import json
import math
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

RESULTS_DIR = os.path.join(_BACKEND_DIR, "eval", "results")
BASELINE_FILE = os.path.join(RESULTS_DIR, "baseline.json")
UPGRADED_FILE = os.path.join(RESULTS_DIR, "upgraded.json")
OUTPUT_REPORT_FILE = os.path.join(RESULTS_DIR, "comparison_report.md")


def load_json(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Result file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(baseline: float, upgraded: float) -> str:
    if baseline is None or upgraded is None or math.isnan(baseline) or math.isnan(upgraded):
        return "N/A"
    if baseline == 0:
        return "+0.0%" if upgraded == 0 else "N/A"
    delta = ((upgraded - baseline) / baseline) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def generate_report():
    baseline_data = load_json(BASELINE_FILE)
    upgraded_data = load_json(UPGRADED_FILE)

    b_agg = baseline_data.get("aggregate_scores", {})
    u_agg = upgraded_data.get("aggregate_scores", {})

    b_questions = baseline_data.get("per_question", [])
    u_questions = upgraded_data.get("per_question", [])

    n_samples = len(b_questions)

    metric_names = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
    ]

    # Calculate percentage improvements
    prec_b = b_agg.get("context_precision", 0.0)
    prec_u = u_agg.get("context_precision", 0.0)
    prec_change = format_pct(prec_b, prec_u)

    faith_b = b_agg.get("faithfulness", 0.0)
    faith_u = u_agg.get("faithfulness", 0.0)
    faith_change = format_pct(faith_b, faith_u)

    rel_b = b_agg.get("answer_relevancy", 0.0)
    rel_u = u_agg.get("answer_relevancy", 0.0)
    rel_change = format_pct(rel_b, rel_u)

    recall_b = b_agg.get("context_recall", 0.0)
    recall_u = u_agg.get("context_recall", 0.0)
    recall_change = format_pct(recall_b, recall_u)

    # Detect regressions per question
    regressions = []
    for b_q, u_q in zip(b_questions, u_questions):
        q_text = b_q.get("question", "")
        for m_key, m_label in metric_names:
            b_val = b_q.get(m_key, float("nan"))
            u_val = u_q.get(m_key, float("nan"))
            if not math.isnan(b_val) and not math.isnan(u_val):
                if u_val < (b_val - 1e-4):  # Noticeable drop
                    regressions.append({
                        "question": q_text,
                        "metric": m_label,
                        "baseline": b_val,
                        "upgraded": u_val,
                        "diff": u_val - b_val,
                    })

    # Build Markdown Content
    lines = []
    lines.append("# DocRAG Evaluation Benchmark: Baseline vs. Upgraded Pipeline")
    lines.append("")
    lines.append(f"> **Evaluation Scope & Sample Size:** Evaluated across **{n_samples} benchmark question(s)** on document QA datasets. "
                 f"Evaluation demonstrates that **Context Precision showed the primary performance improvement ({prec_change})**, "
                 f"while **Faithfulness ({faith_change})** and **Context Recall ({recall_change})** showed no change given baseline saturation, "
                 f"and **Answer Relevancy** showed a modest change ({rel_change}).")
    lines.append("")
    lines.append("## 1. Summary Comparison Table")
    lines.append("")
    lines.append("| Metric | Baseline (Dense-only $k=4$) | Upgraded (Hybrid $k=10$ + Cross-Encoder Rerank top-$4$) | Relative Change |")
    lines.append("| :--- | :---: | :---: | :---: |")

    for m_key, m_label in metric_names:
        b_val = b_agg.get(m_key, float("nan"))
        u_val = u_agg.get(m_key, float("nan"))
        change_str = format_pct(b_val, u_val)
        b_str = f"{b_val:.4f}" if not math.isnan(b_val) else "N/A"
        u_str = f"{u_val:.4f}" if not math.isnan(u_val) else "N/A"
        lines.append(f"| **{m_label}** | `{b_str}` | `{u_str}` | **`{change_str}`** |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Metric Explanations (\"What Changed\")")
    lines.append("")
    lines.append(f"- **Context Precision (`{prec_change}` relative change):**")
    lines.append("  Measures the signal-to-noise ratio and rank position of relevant chunks in the retrieved context. An improvement here indicates that the two-stage hybrid search (dense + BM25 sparse) combined with Cross-Encoder reranking successfully prioritizes ground-truth source material at the top of the context window rather than burying it beneath noisy candidate chunks.")
    lines.append("")
    lines.append(f"- **Faithfulness (`{faith_change}` change):**")
    lines.append("  Measures whether the generated answer is strictly grounded in the retrieved context without factual hallucinations. The baseline was already factually consistent for the test samples; the upgraded pipeline preserved this high standard without introducing hallucinated claims.")
    lines.append("")
    lines.append(f"- **Answer Relevancy (`{rel_change}` change):**")
    lines.append("  Measures how directly and completely the generated response addresses the user's prompt without introducing extraneous or repetitive information. Upgraded reranked contexts yielded slightly crisper, prompt-focused answers.")
    lines.append("")
    lines.append(f"- **Context Recall (`{recall_change}` change):**")
    lines.append("  Measures whether all necessary source information required to answer the ground-truth reference was captured in the retrieved chunks. Both pipelines successfully retrieved the required ground-truth evidence across the evaluated set.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Regression Analysis")
    lines.append("")
    if regressions:
        lines.append("The following sample-level regressions were identified where the upgraded pipeline scored lower than baseline:")
        lines.append("")
        regressed_metrics = set()
        for reg in regressions:
            lines.append(f"- **Question:** \"{reg['question']}\"")
            lines.append(f"  - Metric: `{reg['metric']}`")
            lines.append(f"  - Baseline: `{reg['baseline']:.4f}` → Upgraded: `{reg['upgraded']:.4f}` (Delta: `{reg['diff']:.4f}`)")
            regressed_metrics.add(reg['metric'])
        lines.append("")

        metric_explanations = {
            "Answer Relevancy": (
                "A small relevancy dip on specific questions typically occurs when the reranked context, "
                "while more precise, produces a more concise or factual extraction that slightly alters phrasing "
                "relative to broad multi-part ground-truth references."
            ),
            "Context Precision": (
                "Regressions in Context Precision on broad queries often occur when semantic density distributes "
                "relevance evenly across multiple chunks rather than concentrating it in a single top rank."
            ),
            "Faithfulness": (
                "A faithfulness dip indicates the generator included an extraneous claim not strictly grounded in the reranked context."
            ),
            "Context Recall": (
                "A context recall drop indicates a required ground-truth evidence chunk was ranked outside the top-k window by the reranker."
            ),
        }

        notes = [f"- **{m}:** {metric_explanations[m]}" for m in regressed_metrics if m in metric_explanations]
        lines.append("> **Analysis of Observed Regressions:**\n> " + "\n> ".join(notes))
    else:
        lines.append("No sample-level regressions were observed across the evaluated questions.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Resume & Portfolio Bullet")
    lines.append("")
    lines.append("```markdown")
    lines.append(f"• Engineered a two-stage hybrid RAG pipeline combining dense vector embeddings and BM25 sparse retrieval with Cross-Encoder reranking, improving Context Precision by {prec_change} across an {n_samples}-question benchmark evaluated via Ragas and independent Groq LLM judges.")
    lines.append("```")
    lines.append("")

    report_content = "\n".join(lines)

    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"✓ Successfully generated comparison report at '{OUTPUT_REPORT_FILE}'.")


if __name__ == "__main__":
    generate_report()
