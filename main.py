from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from . import pipeline as rag

app = FastAPI(title="PDF Q&A (LangChain + OpenAI + Qdrant)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    question: str
    top_k: int = 5
    doc_id: Optional[str] = None


def guarded(fn, *a):
    try:
        return fn(*a)
    except RuntimeError as e:
        raise HTTPException(500, str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
def upload(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"filename": f.filename, "error": "Only PDF files are supported"})
            continue
        try:
            results.append(rag.ingest_pdf(f.filename, f.file.read()))
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return {"results": results}


@app.get("/api/documents")
def documents():
    return rag.list_documents()


@app.delete("/api/documents/{doc_id}")
def remove(doc_id: str):
    rag.delete_document(doc_id)
    return {"deleted": doc_id}


@app.post("/api/search")
def search(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "Question is empty")
    return {"results": guarded(rag.search, q.question, q.top_k, q.doc_id)}


@app.post("/api/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "Question is empty")
    hits = guarded(rag.search, q.question, q.top_k, q.doc_id)
    return {"answer": guarded(rag.answer, q.question, hits), "sources": hits}
