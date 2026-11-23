"""3. Knowledge store: FAISS for vectors + SQLite for chunk metadata, citations and query logs.

(SQLite stands in for PostgreSQL so the project runs with zero setup.)
"""

import json
import sqlite3
import time

import faiss
import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,          -- == row position in the FAISS index
    doc TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS query_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    question TEXT,
    answer TEXT,
    confidence TEXT,
    abstained INTEGER,
    supported_ratio REAL,
    sources TEXT,
    feedback INTEGER                 -- +1 helpful, -1 not helpful, NULL none
);
"""


def connect() -> sqlite3.Connection:
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save(chunks: list[dict], vectors: np.ndarray) -> None:
    """Replace the whole knowledge store with these chunks."""
    conn = connect()
    with conn:
        conn.execute("DELETE FROM chunks")
        conn.executemany(
            "INSERT INTO chunks (id, doc, page, section, text) VALUES (?, ?, ?, ?, ?)",
            [(i, c["doc"], c.get("page"), c.get("section"), c["text"]) for i, c in enumerate(chunks)],
        )
    if not chunks:  # knowledge base emptied: drop the vector index too
        config.INDEX_PATH.unlink(missing_ok=True)
        return
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine
    index.add(vectors.astype("float32"))
    faiss.write_index(index, str(config.INDEX_PATH))


class KnowledgeStore:
    def __init__(self):
        self.conn = connect()
        self.reload()

    def reload(self):
        self.index = faiss.read_index(str(config.INDEX_PATH)) if config.INDEX_PATH.exists() else None

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def search(self, qvec: np.ndarray, k: int) -> list[dict]:
        if not self.size:
            return []
        scores, ids = self.index.search(qvec.reshape(1, -1).astype("float32"), min(k, self.size))
        out = []
        for score, cid in zip(scores[0], ids[0]):
            row = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (int(cid),)).fetchone()
            out.append({**dict(row), "score": float(score)})
        return out

    def doc_chunks(self, doc: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM chunks WHERE doc = ? ORDER BY id", (doc,)).fetchall()
        return [{**dict(r), "score": 1.0} for r in rows]

    def clear_history(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM query_logs")

    def documents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT doc, COUNT(*) AS chunks FROM chunks GROUP BY doc ORDER BY doc"
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- 8. logging & analytics ----
    def log_query(self, question: str, result: dict) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO query_logs (ts, question, answer, confidence, abstained, supported_ratio, sources)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    question,
                    result["answer"],
                    result["confidence"],
                    int(result["abstained"]),
                    result["verification"]["supported_ratio"],
                    json.dumps([s["label"] for s in result["sources"]]),
                ),
            )
        return cur.lastrowid

    def set_feedback(self, log_id: int, value: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE query_logs SET feedback = ? WHERE id = ?", (value, log_id))

    def stats(self) -> dict:
        r = self.conn.execute(
            "SELECT COUNT(*) AS queries,"
            " AVG(abstained) AS abstain_rate,"
            " AVG(CASE WHEN abstained = 0 THEN supported_ratio END) AS avg_faithfulness,"
            " SUM(feedback = 1) AS thumbs_up, SUM(feedback = -1) AS thumbs_down"
            " FROM query_logs"
        ).fetchone()
        recent = self.conn.execute(
            "SELECT id, question, confidence, abstained FROM query_logs ORDER BY id DESC LIMIT 10"
        ).fetchall()
        return {**dict(r), "recent": [dict(x) for x in recent]}
