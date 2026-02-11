import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from pydantic import BaseModel
from app.services.ingestion import IngestionService
from app.services.vector_store import VectorStoreService
from app.services.rag_chain import RAGService
from app.core.config import settings

router = APIRouter()

# Services
ingestion_service = IngestionService()
vector_store_service = VectorStoreService()
rag_service = RAGService(vector_store_service)


class ChatRequest(BaseModel):
    question: str
    google_api_key: str | None = None


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    total_docs = 0
    for file in files:
        docs = await ingestion_service.process_file(file)
        vector_store_service.add_documents(docs)
        total_docs += len(docs)
    return {"message": f"Processed {len(files)} files.", "chunks_created": total_docs}


@router.post("/chat")
async def chat(request: ChatRequest):
    api_key = request.google_api_key or settings.GOOGLE_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Google API Key is required.")

    rag_service.update_llm(api_key)

    def generate():
        try:
            for chunk in rag_service.generate_answer(request.question):
                if chunk:
                    yield f"data: {json.dumps({'type': 'answer', 'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/health")
def health_check():
    return {"status": "ok"}
