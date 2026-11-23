"""4-6. Query & retrieval -> generation -> claim verification -> uncertainty -> response."""

import re

import numpy as np

from . import config
from .embeddings import embed
from .store import KnowledgeStore

ABSTAIN_MSG = (
    "I don't have enough information in the knowledge base to answer this reliably. "
    "Please add more details (equipment model, error code, symptoms, when it happens) "
    "or upload the relevant manual."
)

SYSTEM_PROMPT = """You are VeriGuide, a technical troubleshooting assistant.
Answer ONLY using the numbered passages provided. Rules:
- Every sentence must end with the citation(s) of the passage(s) it comes from, e.g. [1] or [2][3].
- Do not add facts, numbers or steps that are not in the passages.
- Be concise: a short diagnosis followed by numbered steps if useful.
- If the passages do not contain the answer, reply with exactly: INSUFFICIENT"""


SUMMARY_PROMPT = """You are VeriGuide. Summarize the document(s) below for a maintenance engineer.
Use ONLY the numbered passages. Give 4-8 bullet points covering the main topics.
Every bullet must end with the citation(s) of the passage(s) it comes from, e.g. [1] or [2][3]."""

# "summarize the file", "give me an overview", "what is in this document", "tl;dr", ...
SUMMARY_RE = re.compile(
    r"\b(?:summar\w*|overview|tl;?dr|key points|main points|gist)\b"
    r"|\bwhat(?:'s| is)? (?:in )?(?:this|the) (?:file|document|doc|manual|pdf)(?: about| contain\w*)?\s*\??\s*$",
    re.I,
)
# Words too generic to identify which file the user means.
GENERIC_WORDS = {"manual", "guide", "file", "document", "doc", "pdf", "txt", "codes", "code", "the", "and"}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*\t") for p in parts if len(p.strip(" -*\t")) > 3]


def strip_citations(s: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", s).strip()


def source_label(p: dict) -> str:
    where = f"p.{p['page']}" if p.get("page") else (f"§ {p['section']}" if p.get("section") else "")
    return f"{p['doc']} {where}".strip()


class VeriGuide:
    def __init__(self):
        self.store = KnowledgeStore()
        self._llm = None
        if config.LLM_API_KEY:
            from openai import OpenAI

            self._llm = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, max_retries=3)

    @property
    def mode(self) -> str:
        return f"LLM ({config.LLM_MODEL})" if self._llm else "offline extractive (no API key)"

    # ---------- 4. Retrieval ----------
    def retrieve(self, question: str, k: int = config.TOP_K) -> list[dict]:
        hits = self.store.search(embed([question])[0], k)
        for n, h in enumerate(hits, start=1):
            h["n"], h["label"] = n, source_label(h)
        return hits

    # ---------- 5a. Generation ----------
    def _generate_llm(self, question: str, passages: list[dict]) -> str:
        context = "\n\n".join(f"[{p['n']}] ({p['label']})\n{p['text']}" for p in passages)
        resp = self._llm.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def _generate_extractive(self, question: str, passages: list[dict]) -> str:
        """Offline fallback: pick the passage sentences most similar to the question."""
        # Only use passages nearly as relevant as the best one, so answers stay on-topic.
        top = passages[0]["score"]
        focus = [p for p in passages if p["score"] >= top - 0.1]
        cands, seen = [], set()
        for p in focus:
            prefix = f"{p['section']}: " if p.get("section") else ""
            for s in split_sentences(p["text"].removeprefix(prefix)):
                if s not in seen:  # overlapping chunks repeat sentences
                    seen.add(s)
                    cands.append((s, p["n"], prefix))
        if not cands:
            return "INSUFFICIENT"
        # Score with the section heading for context, display without it.
        sims = embed([c[2] + c[0] for c in cands]) @ embed([question])[0]
        best = [i for i in np.argsort(-sims)[:4] if sims[i] >= 0.45]
        if not best:
            return "INSUFFICIENT"
        best.sort()  # keep original document order for readability
        return "\n".join(f"- {cands[i][0]} [{cands[i][1]}]" for i in best)

    # ---------- 5b. Claim verification ----------
    def verify(self, answer: str, passages: list[dict]) -> dict:
        claims = [c for c in split_sentences(answer) if strip_citations(c)]
        evidence = [(s, p) for p in passages for s in split_sentences(p["text"])]
        if not claims or not evidence:
            return {"claims": [], "supported_ratio": 0.0}
        claim_vecs = embed([strip_citations(c) for c in claims])
        ev_vecs = embed([e[0] for e in evidence])
        sims = claim_vecs @ ev_vecs.T
        valid_n = {p["n"] for p in passages}
        results = []
        for i, claim in enumerate(claims):
            j = int(np.argmax(sims[i]))
            cited = {int(x) for x in re.findall(r"\[(\d+)\]", claim)}
            supported = bool(sims[i, j] >= config.CLAIM_SUPPORT_THRESHOLD) and cited <= valid_n
            results.append({
                "claim": strip_citations(claim),
                "supported": supported,
                "score": round(float(sims[i, j]), 3),
                "evidence": evidence[j][0],
                "evidence_source": evidence[j][1]["label"],
            })
        ratio = sum(r["supported"] for r in results) / len(results)
        return {"claims": results, "supported_ratio": round(ratio, 3)}

    # ---------- 5c + 6. Uncertainty + final response ----------
    def ask(self, question: str) -> dict:
        if SUMMARY_RE.search(question):
            return self.summarize(question)
        passages = self.retrieve(question)
        top = passages[0]["score"] if passages else 0.0
        base = {"question": question, "mode": self.mode, "top_retrieval_score": round(top, 3)}

        if not passages or top < config.MIN_RETRIEVAL_SCORE:
            return self._abstain(base, passages, "No sufficiently relevant passage was found.")

        answer = (
            self._generate_llm(question, passages) if self._llm
            else self._generate_extractive(question, passages)
        )
        if not answer or answer.upper().startswith("INSUFFICIENT"):
            return self._abstain(base, passages, "The retrieved passages do not contain the answer.")

        verification = self.verify(answer, passages)
        ratio = verification["supported_ratio"]
        if ratio < 0.5:
            return self._abstain(base, passages, "Most claims in the draft answer were not supported by evidence.",
                                 verification)

        # Confidence blends retrieval strength and claim support.
        retrieval_strength = min(1.0, max(0.0, (top - config.MIN_RETRIEVAL_SCORE) / (0.75 - config.MIN_RETRIEVAL_SCORE)))
        score = 0.4 * retrieval_strength + 0.6 * ratio
        confidence = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"

        cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
        sources = [self._src(p) for p in passages if p["n"] in cited] or [self._src(p) for p in passages[:1]]
        warnings = [] if ratio == 1 else [
            f"{sum(not c['supported'] for c in verification['claims'])} claim(s) could not be verified against the sources."
        ]
        return {**base, "answer": answer, "abstained": False, "confidence": confidence,
                "confidence_score": round(score, 3), "sources": sources,
                "verification": verification, "warnings": warnings}

    # ---------- Document summaries ----------
    def target_docs(self, question: str) -> list[str]:
        """Which file(s) the user wants summarized: named in the question, 'all', or the latest upload."""
        docs = [d["doc"] for d in self.store.documents()]
        if not docs:
            return []
        q = question.lower()
        if re.search(r"\b(all|every|each)\b.*\b(files?|documents?|docs?|manuals?)\b", q):
            return docs
        scored = []
        for d in docs:
            words = {w for w in re.split(r"[_\-\s.]+", d.lower()) if len(w) > 2} - GENERIC_WORDS
            hits = sum(w in q for w in words)
            if hits:
                scored.append((hits, d))
        if scored:
            return [max(scored)[1]]
        # "the file" with no name: the most recently added document.
        files = [config.DOCS_DIR / d for d in docs if (config.DOCS_DIR / d).exists()]
        return [max(files, key=lambda p: p.stat().st_mtime).name] if files else docs[:1]

    def summarize(self, question: str) -> dict:
        docs = self.target_docs(question)
        base = {"question": question, "mode": self.mode, "top_retrieval_score": 1.0, "task": "summary"}
        passages = [c for d in docs for c in self.store.doc_chunks(d)]
        for n, p in enumerate(passages, start=1):
            p["n"], p["label"] = n, source_label(p)
        if not passages:
            return self._abstain(base, [], "There is no document to summarize. Upload one first.")

        points_per_doc = 8 if len(docs) == 1 else max(2, 8 // len(docs))
        if self._llm:
            context = "\n\n".join(f"[{p['n']}] ({p['label']})\n{p['text']}" for p in passages)
            resp = self._llm.chat.completions.create(
                model=config.LLM_MODEL, temperature=0,
                messages=[{"role": "system", "content": SUMMARY_PROMPT},
                          {"role": "user", "content": f"Passages:\n{context}\n\nRequest: {question}"}],
            )
            body = (resp.choices[0].message.content or "").strip()
        else:
            body = "\n".join(self._extractive_summary(d, [p for p in passages if p["doc"] == d], points_per_doc)
                             for d in docs)

        verification = self.verify(body, passages)
        ratio = verification["supported_ratio"]
        confidence = "High" if ratio >= 0.9 else "Medium" if ratio >= 0.6 else "Low"
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", body)}
        title = f"Summary of {', '.join(docs)}:\n"
        warnings = [] if ratio == 1 else [
            f"{sum(not c['supported'] for c in verification['claims'])} point(s) could not be verified against the file."
        ]
        return {**base, "answer": title + body, "abstained": False, "confidence": confidence,
                "confidence_score": ratio, "sources": [self._src(p) for p in passages if p["n"] in cited],
                "verification": verification, "warnings": warnings}

    def _extractive_summary(self, doc: str, passages: list[dict], max_points: int) -> str:
        """Offline: from each section/page, keep the sentence most representative of it."""
        groups: dict[str, list[tuple[str, int]]] = {}
        seen = set()
        for p in passages:
            prefix = f"{p['section']}: " if p.get("section") else ""
            key = p.get("section") or f"page {p.get('page')}"
            for s in split_sentences(p["text"].removeprefix(prefix)):
                if s not in seen:
                    seen.add(s)
                    groups.setdefault(key, []).append((s, p["n"]))
        per_group = max(1, max_points // max(1, len(groups)))
        lines = []
        for key, sents in groups.items():
            vecs = embed([s for s, _ in sents])
            centroid = vecs.mean(axis=0)
            best = sorted(np.argsort(-(vecs @ centroid))[:per_group])
            label = "" if key.startswith("page ") else f"{key}: "
            for i in best:
                lines.append(f"- {label}{sents[i][0]} [{sents[i][1]}]")
        return "\n".join(lines[: max_points + 2])

    def _abstain(self, base, passages, reason, verification=None):
        return {**base, "answer": ABSTAIN_MSG, "abstained": True, "confidence": "Low",
                "confidence_score": 0.0, "reason": reason,
                "sources": [self._src(p) for p in passages[:2]],
                "verification": verification or {"claims": [], "supported_ratio": 0.0},
                "warnings": [reason]}

    @staticmethod
    def _src(p: dict) -> dict:
        return {"n": p["n"], "label": p["label"], "doc": p["doc"], "page": p.get("page"),
                "section": p.get("section"), "score": round(p["score"], 3), "text": p["text"]}
