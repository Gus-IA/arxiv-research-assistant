import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from retrieval import hybrid_search, keyword_search, search_and_rerank, vector_search

K = 5

METHODS = {
    "vector": lambda q: vector_search(q, top_k=K),
    "keyword": lambda q: keyword_search(q, top_k=K),
    "hybrid": lambda q: hybrid_search(q, top_k=K),
    "hybrid + rerank": lambda q: search_and_rerank(q, top_k=K),
}


def evaluate(method, questions):
    paper_hits, chunk_hits, rr = 0, 0, []
    for item in questions:
        results = method(item["question"])
        chunk_ids = [r["id"] for r in results]
        paper_hits += item["arxiv_id"] in [r["arxiv_id"] for r in results]
        if item["chunk_id"] in chunk_ids:
            chunk_hits += 1
            rr.append(1 / (chunk_ids.index(item["chunk_id"]) + 1))
        else:
            rr.append(0)
    n = len(questions)
    return paper_hits / n, chunk_hits / n, sum(rr) / n


if __name__ == "__main__":
    with open("data/ground_truth_chunks.json", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"{len(questions)} questions\n")

    print(f"{'method':<18}{'paper hit@5':<14}{'chunk hit@5':<14}{'MRR'}")
    for name, method in tqdm(METHODS.items(), desc="Evaluating"):
        paper_hit, chunk_hit, mrr = evaluate(method, questions)
        tqdm.write(f"{name:<18}{paper_hit:<14.3f}{chunk_hit:<14.3f}{mrr:.3f}")
