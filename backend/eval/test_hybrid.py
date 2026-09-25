"""
eval/test_hybrid.py — Manual retrieval comparison script
=========================================================

Loads the existing chroma_db and BM25 pickle (no uploads needed), runs a
handful of sample queries through three retrieval paths, and prints the top-3
results from each path side-by-side so you can visually spot whether hybrid
is surfacing chunks that dense-only or BM25-only missed.

Run from the backend/ directory:

    python -m eval.test_hybrid

    # or, with an explicit chroma path:
    CHROMA_PERSIST_DIRECTORY=./chroma_db python -m eval.test_hybrid

No assertions, no test framework — pure manual inspection before Ragas.
"""

import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Path hygiene — ensure backend/ is on sys.path so `app.*` imports resolve
# the same way they do when the server runs.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services.vector_store import VectorStoreService  # noqa: E402

# ---------------------------------------------------------------------------
# Sample queries — edit freely.  Aim for variety:
#   • one specific factual question (tests BM25 keyword recall)
#   • one semantic / paraphrase question (tests dense embedding)
#   • one that both should handle well (sanity check for fusion)
#   • one that's deliberately vague (shows graceful degradation)
# ---------------------------------------------------------------------------
QUERIES = [
    "What are the main topics covered in the document?",
    "Explain the key methodology or approach described.",
    "What conclusions or findings are presented?",
    "Are there any limitations or future work mentioned?",
]

TOP_K = 3          # results to print per retriever per query
CANDIDATE_K = 10   # how many each retriever fetches before fusion

# Column widths for the side-by-side table
_COL_W = 72        # width of each result block
_SNIPPET_W = _COL_W - 4  # leave room for the rank prefix


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _snippet(text: str, width: int = _SNIPPET_W, max_lines: int = 4) -> str:
    """Wrap and truncate text to fit neatly in a fixed-width column."""
    wrapped = textwrap.fill(text.strip().replace("\n", " "), width=width)
    lines = wrapped.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 3] + "..."
    return "\n    ".join(lines)  # indent continuation lines by 4 spaces


def _score_line(metadata: dict) -> str:
    """Build a one-line score summary from whatever keys are present."""
    parts = []
    if "hybrid_score" in metadata:
        parts.append(f"hybrid={metadata['hybrid_score']:.4f}")
    if "dense_score_norm" in metadata:
        parts.append(f"dense_n={metadata['dense_score_norm']:.4f}")
    if "bm25_score_norm" in metadata:
        parts.append(f"bm25_n={metadata['bm25_score_norm']:.4f}")
    if "bm25_score" in metadata:
        parts.append(f"bm25_raw={metadata['bm25_score']:.4f}")
    cid = metadata.get("chunk_id", metadata.get("source", "?"))
    return f"[{cid}]  {' | '.join(parts)}" if parts else f"[{cid}]"


def _divider(char: str = "─", width: int = _COL_W * 3 + 8) -> str:
    return char * width


def _header(label: str, width: int = _COL_W) -> str:
    return f"  {'━━ ' + label + ' ━━':^{width - 4}}  "


def _get_doc_key(doc) -> str:
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 0)
    return f"{source}__p{page}__h{abs(hash(doc.page_content)) % 10000}"


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def run(vs: VectorStoreService) -> None:
    print("\n" + "═" * (_COL_W * 3 + 8))
    print("  DocRAG — Retrieval Comparison: Dense  vs  BM25  vs  Hybrid")
    print("═" * (_COL_W * 3 + 8))

    rescued_by_bm25_global = []

    for query_idx, query in enumerate(QUERIES, start=1):
        print(f"\n{'▶':>2}  Query {query_idx}: {query!r}")
        print(_divider())

        # ── Dense (Chroma similarity search) ────────────────────────────────
        raw_dense = vs.vector_db.similarity_search_with_score(query, k=CANDIDATE_K)
        dense_docs = [doc for doc, _ in raw_dense[:TOP_K]]
        dense_scores = {
            _get_doc_key(doc): score
            for doc, score in raw_dense
        }

        # ── BM25 sparse ─────────────────────────────────────────────────────
        bm25_docs = vs.bm25_search(query, k=CANDIDATE_K)[:TOP_K]

        # ── Hybrid (fused) ───────────────────────────────────────────────────
        hybrid_docs = vs.hybrid_search(query, k=CANDIDATE_K)[:TOP_K]

        # ── Side-by-side header row ──────────────────────────────────────────
        print(
            _header("DENSE (distance)", _COL_W)
            + " │ "
            + _header("BM25 (keyword)", _COL_W)
            + " │ "
            + _header("HYBRID (fused)", _COL_W)
        )
        print(_divider("─"))

        # ── Result rows ──────────────────────────────────────────────────────
        for rank in range(TOP_K):
            # Dense column
            if rank < len(dense_docs):
                d_doc = dense_docs[rank]
                d_cid = _get_doc_key(d_doc)
                d_raw_score = dense_scores.get(d_cid, 0.0)
                d_text = _snippet(d_doc.page_content)
                d_score = f"dist_raw={d_raw_score:.4f}  [{d_cid}]"
            else:
                d_text = "(no result)"
                d_score = ""

            # BM25 column
            if rank < len(bm25_docs):
                b_doc = bm25_docs[rank]
                b_text = _snippet(b_doc.page_content)
                b_score = _score_line(b_doc.metadata)
            else:
                b_text = "(no result)"
                b_score = ""

            # Hybrid column
            if rank < len(hybrid_docs):
                h_doc = hybrid_docs[rank]
                h_text = _snippet(h_doc.page_content)
                h_score = _score_line(h_doc.metadata)
            else:
                h_text = "(no result)"
                h_score = ""

            # Print rank label + first snippet line for each column
            rank_label = f"#{rank + 1}"
            print(
                f"  {rank_label} {d_text:<{_COL_W - 4}}"
                f" │   {rank_label} {b_text:<{_COL_W - 4}}"
                f" │   {rank_label} {h_text:<{_COL_W - 4}}"
            )
            # Score line beneath each snippet
            print(
                f"    {d_score:<{_COL_W - 4}}"
                f" │     {b_score:<{_COL_W - 4}}"
                f" │     {h_score:<{_COL_W - 4}}"
            )
            print(_divider("·"))

        # ── Chunk-ID diff: what hybrid added that dense missed ───────────────
        dense_ids  = {_get_doc_key(d) for d in dense_docs}
        bm25_ids   = {_get_doc_key(d) for d in bm25_docs}
        hybrid_ids = {_get_doc_key(d) for d in hybrid_docs}

        only_via_bm25   = hybrid_ids - dense_ids  # hybrid rescued from BM25
        only_via_dense  = hybrid_ids - bm25_ids   # hybrid rescued from dense
        in_both         = hybrid_ids & dense_ids & bm25_ids

        if only_via_bm25:
            rescued_by_bm25_global.append((query_idx, query, sorted(only_via_bm25)))

        print(f"\n  Chunk-ID overlap analysis (top-{TOP_K} results):")
        print(f"    {'In all three lists:':<35} {sorted(in_both) or '(none)'}")
        print(f"    {'Hybrid pulled from BM25 only:':<35} {sorted(only_via_bm25) or '(none)'}")
        print(f"    {'Hybrid pulled from Dense only:':<35} {sorted(only_via_dense) or '(none)'}")
        print()

    print("═" * (_COL_W * 3 + 8))

    # ── Final Summary & Verification Assertions ─────────────────────────────
    chroma_count = vs.vector_db._collection.count()
    bm25_count   = len(vs._bm25_corpus)

    print("\n  SUMMARY & VERIFICATION CHECKS:")
    print(f"    Total chunks in Chroma : {chroma_count}")
    print(f"    Total chunks in BM25   : {bm25_count}")

    # Check 1: Count parity assertion
    count_match = (chroma_count == bm25_count)
    if count_match:
        print(f"    [PASS] Index Count Match: Chroma ({chroma_count}) == BM25 ({bm25_count})")
    else:
        print(f"    [FAIL] Index Count Mismatch: Chroma ({chroma_count}) != BM25 ({bm25_count})")

    # Check 2: BM25 unique contribution
    if rescued_by_bm25_global:
        print("    [PASS] Hybrid Value Addition: BM25 contributed unique chunk(s) not in Dense top-3:")
        for q_idx, q_txt, chunks in rescued_by_bm25_global:
            print(f"           • Query {q_idx} ({q_txt!r}): {chunks}")
    else:
        print("    [INFO] Hybrid Value Addition: Dense and Hybrid top-3 had full overlap for the test queries.")

    print("\n" + "═" * (_COL_W * 3 + 8) + "\n")
    assert count_match, f"Count parity failure: Chroma has {chroma_count} chunks but BM25 has {bm25_count} chunks."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nInitialising VectorStoreService (loading Chroma + BM25 pickle)…")

    vs = VectorStoreService()

    chroma_count = vs.vector_db._collection.count()
    bm25_count   = len(vs._bm25_corpus)

    print(f"  Chroma collection : {chroma_count} vectors")
    print(f"  BM25 corpus       : {bm25_count} chunks")

    if chroma_count == 0 and bm25_count == 0:
        print(
            "\n  ⚠  Both stores are empty — upload at least one document via the\n"
            "     frontend or /api/upload before running this script.\n"
        )
        sys.exit(0)

    if chroma_count == 0:
        print("\n  ⚠  Chroma is empty — dense and hybrid columns will be blank.")
    if bm25_count == 0:
        print("\n  ⚠  BM25 pickle not found — BM25 and hybrid columns will be blank.")

    run(vs)

