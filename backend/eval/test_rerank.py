"""
eval/test_rerank.py — 4-Way Retrieval & Reranking Comparison Script
=====================================================================

Runs test queries through 4 retrieval stages:
  1. Dense (Chroma distance)
  2. Sparse (BM25 score)
  3. Hybrid (Dense + BM25 min-max fused, top-10 candidates)
  4. Reranked (Cross-Encoder ms-marco-MiniLM-L-6-v2, top-4 final)

Prints a 4-column side-by-side comparison table and analyzes position reordering
and new chunk entries brought into top-4 by the cross-encoder.

Run from the backend/ directory:

    python -m eval.test_rerank
"""

import os
import sys
import textwrap

# Path hygiene — ensure backend/ is on sys.path
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services.vector_store import VectorStoreService  # noqa: E402
from app.services.reranker import RerankerService        # noqa: E402

QUERIES = [
    "What are the main topics covered in the document?",
    "Explain the key methodology or approach described.",
    "What conclusions or findings are presented?",
    "Are there any limitations or future work mentioned?",
]

TOP_K = 4          # top results to print per column
CANDIDATE_K = 10   # candidates fetched by hybrid search before reranking

_COL_W = 56        # column width for 4-way table
_SNIPPET_W = _COL_W - 4


def _snippet(text: str, width: int = _SNIPPET_W, max_lines: int = 3) -> str:
    wrapped = textwrap.fill(text.strip().replace("\n", " "), width=width)
    lines = wrapped.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 3] + "..."
    return "\n    ".join(lines)


def _get_doc_key(doc) -> str:
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 0)
    return f"{source}__p{page}__h{abs(hash(doc.page_content)) % 10000}"


def _score_line(doc, raw_dense_map=None) -> str:
    cid = _get_doc_key(doc)
    meta = doc.metadata
    parts = []
    if "rerank_score" in meta:
        parts.append(f"rerank={meta['rerank_score']:.4f}")
    if "hybrid_score" in meta:
        parts.append(f"hybrid={meta['hybrid_score']:.4f}")
    if "bm25_score" in meta:
        parts.append(f"bm25={meta['bm25_score']:.4f}")
    if raw_dense_map and cid in raw_dense_map:
        parts.append(f"dist={raw_dense_map[cid]:.4f}")

    return f"[{cid}]  {' | '.join(parts)}" if parts else f"[{cid}]"


def _divider(char: str = "─", width: int = _COL_W * 4 + 11) -> str:
    return char * width


def _header(label: str, width: int = _COL_W) -> str:
    return f"  {'━━ ' + label + ' ━━':^{width - 4}}  "


def run(vs: VectorStoreService, reranker: RerankerService) -> None:
    print("\n" + "═" * (_COL_W * 4 + 11))
    print("  DocRAG — 4-Stage Retrieval Comparison: Dense vs BM25 vs Hybrid vs Reranked")
    print("═" * (_COL_W * 4 + 11))

    total_reordered = 0
    total_new_entries = 0

    for query_idx, query in enumerate(QUERIES, start=1):
        print(f"\n{'▶':>2}  Query {query_idx}: {query!r}")
        print(_divider())

        # Stage 1a: Dense
        raw_dense = vs.vector_db.similarity_search_with_score(query, k=CANDIDATE_K)
        dense_docs = [doc for doc, _ in raw_dense[:TOP_K]]
        dense_scores = {_get_doc_key(doc): score for doc, score in raw_dense}

        # Stage 1b: BM25
        bm25_docs = vs.bm25_search(query, k=CANDIDATE_K)[:TOP_K]

        # Stage 1c: Hybrid (over-fetch 10 candidates)
        hybrid_candidates = vs.hybrid_search(query, k=CANDIDATE_K)
        hybrid_top4 = hybrid_candidates[:TOP_K]

        # Stage 2: Cross-Encoder Reranking (top-10 -> top-4)
        reranked_top4 = reranker.rerank(query, hybrid_candidates, top_k=TOP_K)

        # Print 4-column headers
        print(
            _header("1. DENSE (dist)", _COL_W)
            + " │ "
            + _header("2. BM25 (sparse)", _COL_W)
            + " │ "
            + _header("3. HYBRID (fused)", _COL_W)
            + " │ "
            + _header("4. RERANKED (cross-enc)", _COL_W)
        )
        print(_divider("─"))

        for rank in range(TOP_K):
            # Dense
            if rank < len(dense_docs):
                d_doc = dense_docs[rank]
                d_cid = _get_doc_key(d_doc)
                d_text = _snippet(d_doc.page_content)
                d_score = f"dist_raw={dense_scores.get(d_cid, 0.0):.4f}  [{d_cid}]"
            else:
                d_text, d_score = "(no result)", ""

            # BM25
            if rank < len(bm25_docs):
                b_doc = bm25_docs[rank]
                b_text = _snippet(b_doc.page_content)
                b_score = _score_line(b_doc)
            else:
                b_text, b_score = "(no result)", ""

            # Hybrid
            if rank < len(hybrid_top4):
                h_doc = hybrid_top4[rank]
                h_text = _snippet(h_doc.page_content)
                h_score = _score_line(h_doc)
            else:
                h_text, h_score = "(no result)", ""

            # Reranked
            if rank < len(reranked_top4):
                r_doc = reranked_top4[rank]
                r_text = _snippet(r_doc.page_content)
                r_score = _score_line(r_doc)
            else:
                r_text, r_score = "(no result)", ""

            rank_label = f"#{rank + 1}"
            print(
                f"  {rank_label} {d_text:<{_COL_W - 4}}"
                f" │   {rank_label} {b_text:<{_COL_W - 4}}"
                f" │   {rank_label} {h_text:<{_COL_W - 4}}"
                f" │   {rank_label} {r_text:<{_COL_W - 4}}"
            )
            print(
                f"    {d_score:<{_COL_W - 4}}"
                f" │     {b_score:<{_COL_W - 4}}"
                f" │     {h_score:<{_COL_W - 4}}"
                f" │     {r_score:<{_COL_W - 4}}"
            )
            print(_divider("·"))

        # Reordering and New Entry Analysis
        hybrid_top4_ids = [_get_doc_key(d) for d in hybrid_top4]
        reranked_top4_ids = [_get_doc_key(d) for d in reranked_top4]

        new_entries = [cid for cid in reranked_top4_ids if cid not in hybrid_top4_ids]

        reordered_count = 0
        for i, cid in enumerate(reranked_top4_ids):
            if i >= len(hybrid_top4_ids) or hybrid_top4_ids[i] != cid:
                reordered_count += 1

        total_reordered += reordered_count
        total_new_entries += len(new_entries)

        print(f"\n  Reranking Impact Analysis (Query {query_idx}):")
        print(f"    • Hybrid top-4 chunk IDs   : {hybrid_top4_ids}")
        print(f"    • Reranked top-4 chunk IDs : {reranked_top4_ids}")
        print(
            f"    • Reordering Summary       : {reordered_count} of {len(reranked_top4_ids)} chunks changed position."
        )
        if new_entries:
            print(f"    • New Candidates Promoted  : {len(new_entries)} chunk(s) from hybrid's top 5-10 entered top-4 -> {new_entries}")
        else:
            print(f"    • New Candidates Promoted  : None (all top-4 reranked chunks came from hybrid's top-4).")
        print()

    print("═" * (_COL_W * 4 + 11))
    print("\n  OVERALL RERANKING SUMMARY:")
    print(f"    • Total Queries Tested           : {len(QUERIES)}")
    print(f"    • Cumulative Position Reorderings: {total_reordered}")
    print(f"    • Cumulative New Chunks Promoted : {total_new_entries}")
    if total_reordered > 0 or total_new_entries > 0:
        print("    • Result                         : [PASS] Reranker actively refined the candidate order.")
    else:
        print("    • Result                         : [NOTICE] Reranker output matched hybrid order for all sample queries.")
    print("\n" + "═" * (_COL_W * 4 + 11) + "\n")


if __name__ == "__main__":
    print("\nInitialising VectorStoreService & RerankerService…")
    vs = VectorStoreService()
    reranker = RerankerService()

    chroma_count = vs.vector_db._collection.count()
    bm25_count = len(vs._bm25_corpus)

    print(f"  Chroma collection : {chroma_count} vectors")
    print(f"  BM25 corpus       : {bm25_count} chunks")

    if chroma_count == 0:
        print("\n  ⚠ Stores are empty. Upload documents before running this script.")
        sys.exit(0)

    run(vs, reranker)
