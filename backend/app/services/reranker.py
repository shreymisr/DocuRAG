from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerService:
    """
    Reranker service using SentenceTransformers CrossEncoder.

    Why Reranking Matters (Recall vs. Precision):
    ---------------------------------------------
    First-stage retrieval (e.g. Hybrid Search combining dense vector similarity
    and BM25 sparse keyword matching) optimizes for **RECALL**. Its job is to
    cast a wide net over thousands/millions of chunks and retrieve a candidate
    set (e.g. top-10 or top-20) with high probability of containing the answer.

    Bi-encoders (like HuggingFaceEmbeddings) compute query and document embeddings
    independently in vector space. While fast for vector search, they miss fine-grained
    token-to-token interactions between the query and passage.

    A **Cross-Encoder** feeds the (query, passage) pair simultaneously into transformer
    cross-attention layers. This allows full attention across query and passage tokens,
    dramatically improving **PRECISION** (reranking accuracy).

    Because full cross-attention is computationally expensive per pair, running it across an
    entire corpus is infeasible. Reranking solves this by running only on the small candidate
    list retrieved in stage 1, refining and sorting them down to a top-k (e.g. k=4)
    for the LLM prompt.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        print(f"[RerankerService] Loading CrossEncoder model '{model_name}'...")
        self.model_name = model_name
        # Load the model ONCE at service initialization (expensive operation)
        self.model = CrossEncoder(self.model_name)
        print(f"[RerankerService] CrossEncoder model '{model_name}' loaded successfully.")

    def rerank(
        self, query: str, chunks: List[Document], top_k: int = 4
    ) -> List[Document]:
        """
        Rerank a candidate list of Document chunks against a query using CrossEncoder.

        Parameters
        ----------
        query : str
            The user's query string.
        chunks : List[Document]
            Candidate Document objects retrieved from first-stage retrieval.
        top_k : int
            Number of top-ranked chunks to return. Defaults to 4.

        Returns
        -------
        List[Document]
            The top_k Document objects sorted descending by relevance score.
            Each Document has 'rerank_score' attached to its metadata dictionary.
        """
        if not chunks:
            return []

        # 1. Build (query, passage) pairs for every chunk passed in
        pairs = [(query, chunk.page_content) for chunk in chunks]

        # 2. Call predict in a single batch call for optimal performance
        scores = self.model.predict(pairs)

        # 3. Attach scores to metadata and pair with chunks
        scored_chunks: List[tuple[Document, float]] = []
        for idx, chunk in enumerate(chunks):
            score = float(scores[idx])
            # Attach rerank_score to metadata for downstream debugging/eval
            updated_metadata = {**chunk.metadata, "rerank_score": round(score, 6)}
            scored_doc = Document(
                page_content=chunk.page_content, metadata=updated_metadata
            )
            scored_chunks.append((scored_doc, score))

        # 4. Sort descending by rerank score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # 5. Return top_k docs
        top_docs = [doc for doc, _ in scored_chunks[:top_k]]
        return top_docs
