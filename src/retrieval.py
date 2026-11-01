import os
import re

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

embedder = SentenceTransformer("all-miniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SELECT_COLS = """
    c.id, c.arxiv_id, c.chunk_index, c.content,
    p.title, p.authors, p.category, p.published, p.pdf_url
"""
FILTERS = """
    AND (%(category)s IS NULL OR p.category = %(category)s)
    AND (%(date_from)s IS NULL OR p.published >= %(date_from)s)
"""


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5433),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def _run(sql, params):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def vector_search(query, top_k=10, category=None, date_from=None):
    """Semantic search: nearest chunks by cosine distance."""
    emb = "[" + ",".join(map(str, embedder.encode(query).tolist())) + "]"
    sql = f"""
        SELECT {SELECT_COLS}
        FROM chunks c JOIN papers p USING (arxiv_id)
        WHERE TRUE {FILTERS}
        ORDER BY c.embedding <=> %(emb)s::vector
        LIMIT %(k)s
    """
    return _run(
        sql, {"emb": emb, "k": top_k, "category": category, "date_from": date_from}
    )


def keyword_search(query, top_k=10, category=None, date_from=None):
    """Full-text search. Terms are joined with OR so long questions still match."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2]
    if not words:
        return []
    tsquery = " | ".join(words)
    sql = f"""
        SELECT {SELECT_COLS}
        FROM chunks c JOIN papers p USING (arxiv_id)
        WHERE c.content_tsv @@ to_tsquery('english', %(q)s) {FILTERS}
        ORDER BY ts_rank(c.content_tsv, to_tsquery('english', %(q)s)) DESC
        LIMIT %(k)s
    """
    return _run(
        sql, {"q": tsquery, "k": top_k, "category": category, "date_from": date_from}
    )


def hybrid_search(query, top_k=10, category=None, date_from=None):
    """Combine vector and keyword rankings with Reciprocal Rank Fusion (RRF)."""
    k_rrf = 60
    scores, docs = {}, {}
    for results in (
        vector_search(query, top_k * 2, category, date_from),
        keyword_search(query, top_k * 2, category, date_from),
    ):
        for rank, doc in enumerate(results):
            docs[doc["id"]] = doc
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k_rrf + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[i] for i in ranked[:top_k]]


def search_and_rerank(query, top_k=5, candidates=20, category=None, date_from=None):
    """Hybrid retrieval followed by a cross-encoder that rescores each (query, chunk) pair."""
    results = hybrid_search(query, candidates, category, date_from)
    if not results:
        return []
    scores = reranker.predict([(query, r["content"]) for r in results])
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda r: r["rerank_score"], reverse=True)[:top_k]


if __name__ == "__main__":
    query = "How do robots learn manipulation from demonstrations?"
    for name, fn in [
        ("Vector", vector_search),
        ("Keyword", keyword_search),
        ("Hybrid", hybrid_search),
    ]:
        print(f"\n=== {name} ===")
        for r in fn(query, top_k=3):
            print(f"[{r['arxiv_id']}] {r['title'][:80]}")
    print("\n=== Hybrid + rerank ===")
    for r in search_and_rerank(query, top_k=3):
        print(f"[{r['arxiv_id']}] ({r['rerank_score']:.2f}) {r['title'][:80]}")
