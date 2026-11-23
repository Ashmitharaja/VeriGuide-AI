"""7. Web application (FastAPI). Run:  uvicorn veriguide.app:app --port 8000"""

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, ingest
from .rag import VeriGuide

app = FastAPI(title="VeriGuide AI")
STATIC = Path(__file__).parent / "static"
vg = VeriGuide()


class Ask(BaseModel):
    question: str


class Feedback(BaseModel):
    log_id: int
    helpful: bool


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"mode": vg.mode, "chunks": vg.store.size, "documents": vg.store.documents()}


@app.post("/api/ask")
def ask(body: Ask):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")
    if not vg.store.size:
        raise HTTPException(400, "Knowledge base is empty. Upload a document or run the ingest step.")
    result = vg.ask(q)
    result["log_id"] = vg.store.log_query(q, result)
    return result


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ingest.SUPPORTED:
        raise HTTPException(400, f"Supported types: {', '.join(sorted(ingest.SUPPORTED))}")
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DOCS_DIR / name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"uploaded": name, "chunks": _reindex()}


def _reindex():
    n = ingest.run()
    vg.store.reload()
    return n


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    path = config.DOCS_DIR / Path(name).name  # basename only: no path traversal
    if not path.is_file():
        raise HTTPException(404, f"No such document: {name}")
    path.unlink()
    return {"deleted": path.name, "chunks": _reindex()}


@app.delete("/api/documents")
def delete_all_documents():
    removed = [p.name for p in config.DOCS_DIR.glob("*") if p.suffix.lower() in ingest.SUPPORTED]
    for name in removed:
        (config.DOCS_DIR / name).unlink()
    _reindex()
    return {"deleted": removed}


@app.delete("/api/history")
def clear_history():
    vg.store.clear_history()
    return {"ok": True}


@app.post("/api/feedback")
def feedback(body: Feedback):
    vg.store.set_feedback(body.log_id, 1 if body.helpful else -1)
    return {"ok": True}


@app.get("/api/stats")
def stats():
    return vg.store.stats()
