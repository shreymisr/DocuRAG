import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastapi import UploadFile


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
            return self.text_splitter.split_documents(documents)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
