from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.core.config import settings


class VectorStoreService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME
        )
        self.vector_db = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=self.embeddings,
        )

    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        self.vector_db.add_documents(documents)

    def get_retriever(self, k: int = 4):
        return self.vector_db.as_retriever(search_kwargs={"k": k})
