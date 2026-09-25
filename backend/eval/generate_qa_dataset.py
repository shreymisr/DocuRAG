"""
eval/generate_qa_dataset.py — QA Dataset Generator using Gemini
================================================================

Reads indexed document chunks from VectorStoreService (ChromaDB), groups them
by source document, and prompts Google Gemini (gemini-2.5-flash) to generate a
draft benchmark Q&A evaluation dataset containing 15-20 total pairs.

Each source document will yield 4-5 pairs:
  - Fact-lookup questions (specific dates, numbers, names, terms)
  - Summary / conceptual questions
  - 1 Unanswerable / out-of-bounds question (for hallucination testing)

Saves output to: backend/eval/qa_dataset_draft.json

Schema:
[
  {
    "question": "...",
    "ground_truth": "...",
    "source_doc": "...",
    "expected_chunk_ids": ["..."]
  }
]

Run from the backend/ directory:
    python -m eval.generate_qa_dataset
"""

import json
import os
import sys
import warnings
from typing import Dict, List

warnings.filterwarnings("ignore", category=FutureWarning)

# Path hygiene — ensure backend/ is on sys.path
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import google.generativeai as genai
from app.core.config import get_active_gemini_model, settings
from app.services.vector_store import VectorStoreService

OUTPUT_DRAFT_PATH = os.path.join(_BACKEND_DIR, "eval", "qa_dataset_draft.json")
PAIRS_PER_DOC = 5

SYSTEM_PROMPT_TEMPLATE = """You are an expert AI evaluation dataset creator for a Retrieval-Augmented Generation (RAG) system.

Given the text chunks from a document titled '{source_doc}', generate exactly {pairs_per_doc} question-and-answer pairs based STRICTLY on the text provided below.

Rules for Question Generation:
1. Diversity of Types:
   - 2 Fact-lookup questions: specific numbers, names, dates, licence numbers, or exact terms present in the text.
   - 2 Summary/conceptual questions: high-level explanations or synthesis of concepts described in the chunks.
   - 1 Unanswerable question: A plausible question about this document domain, but whose answer is NOT present in the provided chunks. For unanswerable questions:
     - Set "ground_truth" to EXACTLY: "I don't have enough information to answer that."
     - Set "expected_chunk_ids" to an empty list [].

2. Precision & Chunk Grounding:
   - For answerable questions, "expected_chunk_ids" MUST list the exact chunk_id(s) where the answer was found (e.g. ["doc.pdf__p0__c0"]).
   - Ground truth answers must be concise, accurate, and completely supported by the text.

3. Output Format:
   - Return ONLY a valid JSON array of objects. Do not include markdown code blocks (```json) or conversational text.
   - Match this JSON structure strictly:
[
  {{
    "question": "What is the licence number?",
    "ground_truth": "The licence number is GJ02 /0002507/2026.",
    "source_doc": "{source_doc}",
    "expected_chunk_ids": ["{source_doc}__p0__c0"]
  }}
]

Here are the labeled chunks from the document:
==================================================
{chunks_formatted}
==================================================

Generate the {pairs_per_doc} QA objects as JSON array now:"""


def fetch_chunks_by_document(vs: VectorStoreService) -> Dict[str, List[dict]]:
    """Group all stored chunks in ChromaDB by their source document."""
    data = vs.vector_db.get()
    docs_text = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    grouped: Dict[str, List[dict]] = {}
    for i, text in enumerate(docs_text):
        meta = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}
        source = meta.get("source") or meta.get("file_name") or "unknown_doc"
        cid = meta.get("chunk_id", f"{source}__c{i}")
        meta["chunk_id"] = cid

        if source not in grouped:
            grouped[source] = []
        grouped[source].append({"chunk_id": cid, "content": text, "metadata": meta})

    return grouped


def main():
    api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is not set in .env or environment.")
        sys.exit(1)

    model_name = get_active_gemini_model(api_key)
    model = genai.GenerativeModel(model_name)

    print("\nInitialising VectorStoreService to load document chunks...")
    vs = VectorStoreService()
    docs_map = fetch_chunks_by_document(vs)

    if not docs_map:
        print("No documents found in ChromaDB. Upload documents before running dataset generator.")
        sys.exit(0)

    print(f"Found {len(docs_map)} document(s) in ChromaDB:")
    for doc_name, chunks in docs_map.items():
        print(f"  • {doc_name}: {len(chunks)} chunk(s)")

    dataset_draft: List[dict] = []

    for source_doc, chunks in docs_map.items():
        print(f"\nGenerating QA pairs for '{source_doc}' using Gemini...")

        # Format labeled chunks for the prompt
        chunks_formatted = "\n\n".join(
            f"--- CHUNK ID: {c['chunk_id']} ---\n{c['content']}" for c in chunks
        )

        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            source_doc=source_doc,
            pairs_per_doc=PAIRS_PER_DOC,
            chunks_formatted=chunks_formatted,
        )

        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()

            # Clean JSON markdown fences if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            qa_pairs = json.loads(raw_text)

            # Validate and enrich each pair
            for pair in qa_pairs:
                pair.setdefault("source_doc", source_doc)
                if not isinstance(pair.get("expected_chunk_ids"), list):
                    pair["expected_chunk_ids"] = []
                dataset_draft.append(pair)

            print(f"  ✓ Generated {len(qa_pairs)} pairs for '{source_doc}'.")

        except Exception as e:
            print(f"  ✗ Error generating QA pairs for '{source_doc}': {e}")

    # Save to JSON draft file
    os.makedirs(os.path.dirname(OUTPUT_DRAFT_PATH), exist_ok=True)
    with open(OUTPUT_DRAFT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_draft, f, indent=2, ensure_ascii=False)

    print(f"\n" + "═" * 80)
    print(f"Successfully generated {len(dataset_draft)} QA pair(s) in total.")
    print(f"Saved draft to: {OUTPUT_DRAFT_PATH}")
    print("═" * 80)

    # Print draft JSON to console for manual review
    print("\n--- DRAFT DATASET FOR REVIEW ---")
    print(json.dumps(dataset_draft, indent=2, ensure_ascii=False))
    print("--- END OF DRAFT DATASET ---\n")


if __name__ == "__main__":
    main()
