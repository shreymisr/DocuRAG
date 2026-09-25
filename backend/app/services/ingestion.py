import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastapi import UploadFile


def _stamp_chunk_ids(chunks: List[Document], source_name: str) -> List[Document]:
    """
    Mutate each chunk's metadata in-place to add a stable ``chunk_id``.

    The ID format is::

        <source_name>__p<page>__c<chunk_index>

    where ``<page>`` is the zero-based page number taken from
    ``doc.metadata.get('page', 0)`` (PyPDFLoader sets this; TextLoader does
    not, so it defaults to 0), and ``<chunk_index>`` is the chunk's position
    within the full chunk list for this file.

    Both Chroma and BM25 receive the *same* Document objects (not copies),
    so modifying metadata here propagates to both stores automatically.
    """
    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        chunk.metadata["chunk_id"] = f"{source_name}__p{page}__c{i}"
        # Ensure source is always present for traceability.
        chunk.metadata.setdefault("source", source_name)
    return chunks


class IngestionService:
    def __init__(self, upload_dir: str = "temp_uploads"):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    async def process_file(self, file: UploadFile) -> List[Document]:
        file_path = os.path.join(self.upload_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            if file.filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.filename.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file.filename}")

            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            # Stamp chunk_id onto every chunk *before* returning so that
            # both Chroma and BM25 share the exact same Document objects
            # with identical metadata — single source of truth for IDs.
            return _stamp_chunk_ids(chunks, source_name=file.filename)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
