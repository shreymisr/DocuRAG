import google.generativeai as genai
from app.services.vector_store import VectorStoreService
from app.services.reranker import RerankerService
from app.core.config import get_active_gemini_model, settings


PROMPT_TEMPLATE = """Answer the question based only on the following context.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        reranker_service: RerankerService | None = None,
    ):
        self.vector_store_service = vector_store_service
        self.reranker_service = reranker_service or RerankerService()
        self.api_key = settings.GOOGLE_API_KEY
        self.model = None
        if self.api_key:
            self._init_llm()

    def _init_llm(self):
        genai.configure(api_key=self.api_key)
        model_name = get_active_gemini_model(self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def update_llm(self, api_key: str):
        if api_key and api_key != self.api_key:
            self.api_key = api_key
            self._init_llm()

    def get_context(
        self, question: str, candidate_k: int = 10, top_k: int = 4
    ) -> str:
        """Retrieve candidates via hybrid search, rerank with CrossEncoder, and format context.

        1. Stage-1 (Recall): Fetches candidate_k (default 10) candidates via hybrid search.
        2. Stage-2 (Precision): Reranks candidates using CrossEncoder down to top_k (default 4).
        3. Formats the final top_k reranked chunks into the prompt context string.

        Parameters
        ----------
        question : str
            The user's natural-language question.
        candidate_k : int
            Number of candidate chunks to fetch from hybrid search (default 10).
        top_k : int
            Number of final reranked chunks to retain for the context (default 4).

        Returns
        -------
        str
            Chunk texts joined by double newlines, ready to drop into the prompt.
        """
        # Step 1: Hybrid search (dense + sparse recall)
        candidates = self.vector_store_service.hybrid_search(
            question, k=candidate_k
        )
        if not candidates:
            return ""

        # Step 2: Cross-Encoder reranking (precision)
        reranked = self.reranker_service.rerank(
            question, candidates, top_k=top_k
        )
        if not reranked:
            return ""

        # Step 3: Format reranked top-4 chunks into context
        return "\n\n".join(doc.page_content for doc in reranked)

    def generate_answer(self, question: str):
        """Generate an answer using retrieved context + Gemini."""
        if not self.model:
            raise ValueError("LLM not initialized. Provide a Google API Key.")

        context = self.get_context(question)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = self.model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

