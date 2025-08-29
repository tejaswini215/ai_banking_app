"""
faq_loader.py

Implements Retrieval for FAQs. Behavior:
- If OPENAI_API_KEY is in env, uses OpenAI embeddings for vector store + optional LLM generation.
- Falls back to TF-IDF + cosine similarity if no API key.
- If FAISS is installed, uses FAISS for nearest neighbor search.
"""

import os
import json
from typing import List, Tuple
import numpy as np

# TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional packages
try:
    import openai
    from sklearn.neighbors import NearestNeighbors
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# FAISS optional
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

class FAQRetriever:
    def __init__(self, faq_json_path: str = None):
        self.faqs = []
        self.questions = []
        self.answers = []
        self.faq_json_path = faq_json_path
        self.use_openai = False
        self.embeddings = None
        self.nn = None
        self.vectorizer = None
        self.doc_matrix = None

        if faq_json_path and os.path.exists(faq_json_path):
            with open(faq_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if q and a:
                    self.faqs.append({"q": q, "a": a})
                    self.questions.append(q)
                    self.answers.append(a)

        # if OpenAI key present, configure
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        if api_key and OPENAI_AVAILABLE:
            openai.api_key = api_key
            self.use_openai = True
            self._build_openai_index()
        else:
            self._build_tfidf_index()

    # -------- TF-IDF fallback ----------
    def _build_tfidf_index(self):
        if len(self.questions) == 0:
            self.vectorizer = None
            self.doc_matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.questions)

    def _tfidf_query(self, query: str, top_k: int = 3):
        if not self.vectorizer or not query.strip():
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.doc_matrix).flatten()
        idxs = np.argsort(-sims)[:top_k]
        results = []
        for i in idxs:
            results.append((self.questions[i], self.answers[i], float(sims[i])))
        return results

    # -------- OpenAI embeddings index ----------
    def _build_openai_index(self):
        if len(self.questions) == 0:
            return
        # get embeddings for questions
        # NOTE: choose the model you have access to (text-embedding-3-small / text-embedding-3-large). Adjust if needed.
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        vecs = []
        for q in self.questions:
            resp = openai.Embedding.create(input=q, model=model)
            vecs.append(resp["data"][0]["embedding"])
        self.embeddings = np.array(vecs).astype("float32")
        # build nearest neighbor index
        if FAISS_AVAILABLE:
            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        else:
            # sklearn NearestNeighbors with cosine (angular distance)
            self.nn = NearestNeighbors(n_neighbors=min(10, len(self.embeddings)), metric="cosine").fit(self.embeddings)

    def _openai_query(self, query: str, top_k: int = 3):
        if not self.use_openai:
            return []
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        resp = openai.Embedding.create(input=query, model=model)
        qv = np.array(resp["data"][0]["embedding"]).astype("float32")
        if FAISS_AVAILABLE:
            faiss.normalize_L2(qv.reshape(1, -1))
            D, I = self.index.search(qv.reshape(1, -1), top_k)
            results = []
            for idx, score in zip(I[0], D[0]):
                results.append((self.questions[idx], self.answers[idx], float(score)))
            return results
        else:
            # sklearn nearest neighbors: note metric is cosine, returns distances
            dists, idxs = self.nn.kneighbors(qv.reshape(1, -1), n_neighbors=min(top_k, len(self.questions)))
            results = []
            for dist, idx in zip(dists[0], idxs[0]):
                score = 1.0 - dist  # rough convert
                results.append((self.questions[idx], self.answers[idx], float(score)))
            return results

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Return list of (question, answer, score)."""
        if self.use_openai:
            return self._openai_query(query, top_k=top_k)
        else:
            return self._tfidf_query(query, top_k=top_k)
