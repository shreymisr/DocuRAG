import google.generativeai as genai
from app.services.vector_store import VectorStoreService
from app.core.config import settings


PROMPT_TEMPLATE = """Answer the question based only on the following context.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        self.api_key = settings.GOOGLE_API_KEY
        self.model = None
        if self.api_key:
            self._init_llm()

    def _init_llm(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def update_llm(self, api_key: str):
        if api_key and api_key != self.api_key:
            self.api_key = api_key
            self._init_llm()

    def get_context(self, question: str) -> str:
        """Retrieve relevant documents and format them as context."""
        retriever = self.vector_store_service.get_retriever()
        docs = retriever.invoke(question)
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)

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
