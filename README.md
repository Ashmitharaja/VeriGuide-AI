# VeriGuide AI — Evidence-Grounded Troubleshooting Assistant

A small, working implementation of the VeriGuide block diagram: ask a troubleshooting question in
plain language and get an answer **built only from your manuals**, with **source citations**,
**claim-level verification**, a **confidence score**, and an **abstain** response when the
documents don't cover the question.

## How it maps to the diagram

| Diagram block | Where it lives |
|---|---|
| 1. Input layer (PDF, manuals, docs) | `data/docs/` (`.pdf`, `.md`, `.txt`) or upload in the UI |
| 2. Processing: parsing/OCR → chunking → embeddings | `veriguide/ingest.py`, `veriguide/embeddings.py` (all-MiniLM-L6-v2) |
| 3. Knowledge store: vector DB + relational DB | `veriguide/store.py` — FAISS + SQLite (SQLite stands in for PostgreSQL: no setup) |
| 4. Query & retrieval (semantic search, top-k) | `VeriGuide.retrieve` in `veriguide/rag.py` |
| 5. Generation, claim verification, uncertainty | `VeriGuide.ask` / `verify` in `veriguide/rag.py` |
| 6. Response (answer + citations + confidence) | JSON from `POST /api/ask` |
| 7. Web UI | `veriguide/app.py` (FastAPI) + `veriguide/static/index.html` |
| 8. Evaluation & monitoring | `veriguide/evaluate.py`, query logs + feedback in SQLite, `/api/stats` |

**Verification logic:** each sentence of the answer is embedded and compared with every sentence of
the retrieved passages. A claim is *supported* if its best match is ≥ 0.55 cosine similarity and
its `[n]` citations point to real passages. The system abstains if the best passage scores
< 0.35, if the LLM says the passages lack the answer, or if under half the claims are supported.
Confidence = 40% retrieval strength + 60% supported-claim ratio → High / Medium / Low.

## Steps to run (Windows PowerShell)

**1. Go to the project folder**
```powershell
cd C:\Users\ashmi\Downloads\VeriGuide
```

**2. Create a virtual environment and install dependencies** (first install downloads PyTorch, a few minutes)
```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\Activate.ps1
```
```powershell
pip install -r requirements.txt
```

**3. (Optional) Add an LLM key.** Without one, VeriGuide still works in *offline extractive* mode
(it quotes the most relevant sentences). With a key, an LLM writes the answer. Free Gemini key:
https://aistudio.google.com/apikey
```powershell
$env:GEMINI_API_KEY = "your-key"
```
Any OpenAI-compatible endpoint also works, e.g. a local Ollama model:
```powershell
$env:LLM_BASE_URL = "http://localhost:11434/v1"; $env:LLM_API_KEY = "ollama"; $env:LLM_MODEL = "qwen2.5:3b"
```

**4. Index the documents** in `data/docs/` (downloads the ~90 MB embedding model the first time)
```powershell
python -m veriguide.ingest
```

**5. Start the web app**
```powershell
uvicorn veriguide.app:app --port 8000
```
Open http://localhost:8000, click an example question or type your own, and upload more
PDF/MD/TXT manuals from the sidebar (they're indexed immediately).

**6. Run the evaluation** (accuracy, retrieval hit rate, faithfulness, hallucination rate, abstention)
```powershell
python -m veriguide.evaluate
```
Results are also written to `data/store/eval_results.json`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ask` | `{"question": "..."}` → answer, sources, confidence, verification |
| POST | `/api/upload` | multipart `file` → saves to `data/docs/` and re-indexes |
| POST | `/api/feedback` | `{"log_id": 1, "helpful": true}` |
| GET | `/api/status` | mode, indexed documents |
| GET | `/api/stats` | query count, abstain rate, avg faithfulness, feedback |
| DELETE | `/api/documents/{name}` | delete one file and re-index |
| DELETE | `/api/documents` | delete all files (empty knowledge base) |
| DELETE | `/api/history` | clear question history (used by **+ New chat**) |

## Using the web page

- **Ask** any troubleshooting question, or click an example under *Try*.
- **Summarize a file:** type `summarize the file` (summarizes the most recently uploaded file),
  `summarize the pump guide` (picks the file by name), or `summarize all files`. Every bullet is
  cited and verified like a normal answer.
- **+ New chat** (top right) clears the conversation and question history.
- **✕** next to a file deletes it; **Remove all files** empties the knowledge base, so you can
  start fresh with your own uploads.

## Notes and limits

- **OCR:** text PDFs work out of the box. For scanned PDFs, `pip install pytesseract pdf2image`
  and install Tesseract + Poppler; empty pages are then OCR'd automatically.
- **Offline mode** can't tell when retrieved text is related but doesn't answer the question
  (e.g. "warranty period" gets IM-300 specs instead of an abstain). LLM mode handles this by
  replying `INSUFFICIENT`, which triggers the abstain path.
- Thresholds, chunk size and top-k are in `veriguide/config.py` (each overridable by env var).
- Low-memory machines: if you see `OpenBLAS error: Memory allocation`, run
  `$env:OPENBLAS_NUM_THREADS = "1"` before the commands.
