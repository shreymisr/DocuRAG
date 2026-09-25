import os
import pickle
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.config import settings

# Filename for the persisted BM25 corpus, stored next to the Chroma directory.
_BM25_PICKLE_FILENAME = "bm25_corpus.pkl"

# Default dense-vs-sparse balance for hybrid_search.
# alpha=1.0 → pure dense (Chroma cosine similarity only)
# alpha=0.0 → pure sparse (BM25 only)
# alpha=0.5 → equal weighting (recommended starting point)
# Tune this constant without changing any call-sites or method signatures.
DEFAULT_HYBRID_ALPHA: float = 0.5


def _tokenize(text: str) -> List[str]:
    """Lowercase + whitespace split — fast, no heavy NLP dependencies."""
    return text.lower().split()


class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        self.vector_db = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
        )

        # --- BM25 state --------------------------------------------------
        # _bm25_corpus holds a list of dicts: {"content": str, "metadata": dict}
        # so we can reconstruct LangChain Document objects after a BM25 search.
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: List[dict] = []

        # Attempt to restore a previously persisted corpus from disk.
        self._bm25_pickle_path = os.path.join(
            settings.CHROMA_PERSIST_DIRECTORY, _BM25_PICKLE_FILENAME
        )
        self._load_bm25_from_disk()
        self._sync_bm25_from_chroma_if_needed()

    # ------------------------------------------------------------------
    # Dense retrieval (existing) — untouched
    # ------------------------------------------------------------------

    def add_documents(self, documents: List[Document]):
        """Add documents to Chroma AND update the BM25 index."""
        if not documents:
            return
        self.vector_db.add_documents(documents)
        # Keep BM25 in sync with every new batch of chunks.
        self.build_bm25_index(documents, extend=True)

    def get_retriever(self, k: int = 4):
        """Return a LangChain retriever backed by Chroma dense similarity search."""
        return self.vector_db.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return top-k Document chunks using dense similarity search in Chroma."""
        return self.vector_db.similarity_search(query, k=k)

    # ------------------------------------------------------------------
    # BM25 sparse retrieval (new)
    # ------------------------------------------------------------------

    def build_bm25_index(
        self, chunks: List[Document], extend: bool = False
    ) -> None:
        """
        Build (or extend) a BM25Okapi index from the given Document chunks.

        **Extend-from-disk contract** — When the service starts up,
        ``_load_bm25_from_disk`` restores any previously persisted corpus into
        ``self._bm25_corpus``.  Callers that pass ``extend=True`` (including
        ``add_documents``) therefore *always* append to whatever is already in
        memory — whether that state came from a previous session's pickle or
        from chunks added earlier in the current session.  This means:

        * A fresh container with no pickle → corpus starts empty, chunks are
          appended → pickle is written after the first upload.
        * A restarted container with an existing pickle → corpus is restored at
          ``__init__`` time; subsequent uploads extend it in memory *and* on
          disk, never rebuilding from scratch.

        Parameters
        ----------
        chunks : list of Document
            The chunks whose ``page_content`` will be indexed.  Every chunk
            **must** already have a ``chunk_id`` key in its ``metadata``
            (stamped by ``IngestionService._stamp_chunk_ids``) so that BM25
            results are traceable to the same source of truth as Chroma results.
        extend : bool
            If ``True``, append new chunks to the existing corpus (default for
            ``add_documents``).  If ``False``, replace the corpus entirely —
            only use this for explicit full rebuilds.
        """
        if not chunks:
            return

        # Enforce the single-source-of-truth invariant: every chunk must carry
        # a chunk_id so BM25 results can be correlated with Chroma results.
        missing_ids = [
            i for i, doc in enumerate(chunks)
            if "chunk_id" not in doc.metadata
        ]
        if missing_ids:
            raise ValueError(
                f"build_bm25_index: chunks at positions {missing_ids} are "
                "missing 'chunk_id' in metadata. Ensure IngestionService "
                "stamps IDs before calling this method."
            )

        new_entries = [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),  # shallow copy
            }
            for doc in chunks
        ]

        if extend:
            self._bm25_corpus.extend(new_entries)
        else:
            self._bm25_corpus = new_entries

        # Tokenize the full (possibly extended) corpus and rebuild the index.
        tokenized_corpus = [_tokenize(entry["content"]) for entry in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        print(
            f"[VectorStoreService] Built BM25 index with {len(self._bm25_corpus)} chunks."
        )

        # Persist to disk — always reflects the full cumulative corpus,
        # so the next container restart will load everything at once.
        self._save_bm25_to_disk()

    def bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Return the top-k chunks by BM25 score as LangChain Document objects.

        The returned Documents have the same schema as Chroma results:
        ``doc.page_content`` holds the text and ``doc.metadata`` holds the
        original metadata dict (including ``chunk_id`` when present).

        Parameters
        ----------
        query : str
            The user's natural-language question.
        k : int
            Maximum number of results to return.

        Returns
        -------
        list of Document
            Ranked from highest BM25 score to lowest.
        """
        if self._bm25_index is None or not self._bm25_corpus:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # Pair each corpus entry with its score, sort descending, take top-k.
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results: List[Document] = []
        for idx, score in ranked:
            entry = self._bm25_corpus[idx]
            # Mirror metadata structure and attach BM25 score for traceability.
            metadata = {**entry["metadata"], "bm25_score": float(score)}
            results.append(Document(page_content=entry["content"], metadata=metadata))

        return results

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = DEFAULT_HYBRID_ALPHA,
    ) -> List[Document]:
        """
        Retrieve the top-k chunks using a weighted fusion of dense (Chroma)
        and sparse (BM25) scores — commonly called "hybrid search".

        **Why min-max normalization is necessary here**

        BM25 scores and Chroma distance scores live on completely different
        numerical scales and directions:

        * Chroma returns distance scores (lower is better, 0.0 = exact match).
        * BM25 returns term-frequency scores (higher is better, 0.0 = no match).

        We convert Chroma's lower-is-better distance to a higher-is-better
        similarity score (similarity = 1 / (1 + distance)) before min-max
        normalization. Min-max normalization rescales each candidate set
        independently to [0, 1] before fusion, making the two scores directly
        comparable and ensuring ``alpha`` actually controls the balance as intended.

        **Fusion formula (Reciprocal Score Fusion variant)**

        For each unique chunk (matched by ``chunk_id``)::

            final_score = alpha * dense_norm + (1 - alpha) * bm25_norm

        Chunks that appear in only one retriever's candidate list receive a
        normalized score from that retriever and 0.0 from the other.

        Parameters
        ----------
        query : str
            The user's natural-language question.
        k : int
            Number of final results to return after fusion and deduplication.
        alpha : float
            Weight given to the dense (Chroma) score. Must be in [0.0, 1.0].
            ``1 - alpha`` is the weight given to the BM25 score.
            Defaults to ``DEFAULT_HYBRID_ALPHA`` (module-level constant).

        Returns
        -------
        list of Document
            Top-k chunks sorted by ``final_score`` (descending). Each
            Document's metadata contains the original fields from ingestion
            (including ``chunk_id``) plus ``dense_score_norm``,
            ``bm25_score_norm``, and ``hybrid_score`` for inspection.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha!r}")

        candidate_k = k * 2  # over-fetch so fusion has enough material

        # ── 1. Dense candidates ─────────────────────────────────────────────
        # similarity_search_with_score returns (Document, distance) tuples where lower = better.
        raw_dense: list[tuple[Document, float]] = (
            self.vector_db.similarity_search_with_score(query, k=candidate_k)
        )

        # ── 2. Sparse (BM25) candidates ─────────────────────────────────────
        raw_bm25: List[Document] = self.bm25_search(query, k=candidate_k)

        # ── 3. Build score maps keyed by chunk_id ───────────────────────────
        def _doc_key(doc: Document) -> str:
            cid = doc.metadata.get("chunk_id")
            if cid:
                return str(cid)
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            return f"{source}__p{page}__h{abs(hash(doc.page_content)) % 10000}"

        dense_scores: dict[str, float] = {}
        dense_docs: dict[str, Document] = {}
        for doc, score in raw_dense:
            cid = _doc_key(doc)
            distance = float(score)
            # Convert lower-is-better distance to higher-is-better similarity
            similarity = 1.0 / (1.0 + max(0.0, distance))
            dense_scores[cid] = similarity
            dense_docs[cid] = doc

        bm25_scores: dict[str, float] = {}
        bm25_docs: dict[str, Document] = {}
        for doc in raw_bm25:
            cid = _doc_key(doc)
            # bm25_score was attached by bm25_search; fall back to 0 if absent.
            bm25_scores[cid] = float(doc.metadata.get("bm25_score", 0.0))
            bm25_docs[cid] = doc

        # ── 4. Min-max normalize each score set independently ───────────────
        dense_norm = self._minmax_normalize(dense_scores)
        bm25_norm = self._minmax_normalize(bm25_scores)

        # ── 5. Fuse scores over the union of all candidate chunk IDs ────────
        all_ids = set(dense_norm) | set(bm25_norm)
        fused: list[tuple[str, float, float, float]] = []  # (cid, d, b, hybrid)
        for cid in all_ids:
            d = dense_norm.get(cid, 0.0)
            b = bm25_norm.get(cid, 0.0)
            hybrid = alpha * d + (1.0 - alpha) * b
            fused.append((cid, d, b, hybrid))

        # Sort descending by hybrid score, take top-k.
        fused.sort(key=lambda x: x[3], reverse=True)
        fused = fused[:k]

        # ── 6. Reconstruct Document objects with enriched metadata ──────────
        results: List[Document] = []
        for cid, d_norm, b_norm, hybrid_score in fused:
            # Prefer the dense doc (richer metadata from Chroma); fall back to
            # the BM25 doc if it only appeared in the sparse results.
            base_doc = dense_docs.get(cid) or bm25_docs[cid]
            metadata = {
                **base_doc.metadata,
                "dense_score_norm": round(d_norm, 6),
                "bm25_score_norm": round(b_norm, 6),
                "hybrid_score": round(hybrid_score, 6),
            }
            results.append(
                Document(page_content=base_doc.page_content, metadata=metadata)
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers — disk persistence & sync
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax_normalize(scores: dict) -> dict:
        """
        Rescale a dict of {id: raw_score} to {id: normalized_score} in [0, 1].

        If all scores are equal (including the degenerate single-item case),
        every entry is mapped to 1.0 so it still participates in fusion rather
        than collapsing to 0.0 (which would incorrectly suppress good results).
        """
        if not scores:
            return {}
        lo = min(scores.values())
        hi = max(scores.values())
        span = hi - lo
        if span == 0.0:
            # All scores identical — treat every candidate as maximally relevant
            # on this axis rather than zeroing them all out.
            return {cid: 1.0 for cid in scores}
        return {cid: (v - lo) / span for cid, v in scores.items()}

    def _sync_bm25_from_chroma_if_needed(self) -> None:
        """If BM25 corpus is empty but Chroma contains vectors, auto-sync BM25 from Chroma."""
        if self._bm25_corpus:
            return
        try:
            collection_count = self.vector_db._collection.count()
            if collection_count > 0:
                print(
                    f"[VectorStoreService] BM25 corpus empty but Chroma contains {collection_count} vectors. Auto-syncing BM25 index..."
                )
                data = self.vector_db.get()
                docs_text = data.get("documents") or []
                metadatas = data.get("metadatas") or []
                recovered_docs = []
                for i, text in enumerate(docs_text):
                    meta = (
                        dict(metadatas[i])
                        if i < len(metadatas) and metadatas[i]
                        else {}
                    )
                    if "chunk_id" not in meta:
                        source = meta.get("source", "chroma")
                        page = meta.get("page", 0)
                        meta["chunk_id"] = f"{source}__p{page}__c{i}"
                    recovered_docs.append(
                        Document(page_content=text, metadata=meta)
                    )
                if recovered_docs:
                    self.build_bm25_index(recovered_docs, extend=False)
        except Exception as exc:  # pragma: no cover
            print(
                f"[VectorStoreService] Warning: could not auto-sync BM25 from Chroma: {exc}"
            )

    def _save_bm25_to_disk(self) -> None:
        """Persist the BM25 corpus list to a pickle file next to Chroma."""
        try:
            # Ensure the directory exists (it will if Chroma already ran, but
            # be safe in case BM25 is built before Chroma writes anything).
            os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            abs_path = os.path.abspath(self._bm25_pickle_path)
            with open(self._bm25_pickle_path, "wb") as fh:
                pickle.dump(self._bm25_corpus, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[VectorStoreService] Persisted BM25 corpus ({len(self._bm25_corpus)} chunks) to '{abs_path}'."
            )
        except Exception as exc:  # pragma: no cover
            # Persistence failure should not crash the request pipeline.
            print(f"[VectorStoreService] Warning: could not save BM25 corpus: {exc}")

    def _load_bm25_from_disk(self) -> None:
        """Restore a previously persisted BM25 corpus and rebuild the index."""
        abs_path = os.path.abspath(self._bm25_pickle_path)
        if not os.path.exists(self._bm25_pickle_path):
            print(f"[VectorStoreService] No BM25 pickle found at '{abs_path}'.")
            return
        try:
            with open(self._bm25_pickle_path, "rb") as fh:
                corpus = pickle.load(fh)
            if corpus:
                self._bm25_corpus = corpus
                tokenized_corpus = [
                    _tokenize(entry["content"]) for entry in self._bm25_corpus
                ]
                self._bm25_index = BM25Okapi(tokenized_corpus)
                print(
                    f"[VectorStoreService] Loaded BM25 corpus ({len(self._bm25_corpus)} chunks) from '{abs_path}'."
                )
        except Exception as exc:  # pragma: no cover
            print(f"[VectorStoreService] Warning: could not load BM25 corpus: {exc}")

