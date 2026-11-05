import json
import os
import sys
import time

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from retrieval import get_conn

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


OUTPUT = "data/ground_truth_chunks.json"

PROMPT = (
    "Write ONE specific question that this excerpt from a scientific paper answers. "
    "Do NOT mention the paper title, method names, or phrases like 'the excerpt' or 'the paper'. "
    "Return only the question.\n\nExcerpt:\n{text}"
)


def load_existing():
    """Load questions saved in a previous run so we can resume."""
    if os.path.exists(OUTPUT):
        try:
            with open(OUTPUT, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


def save(questions):
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)


def main():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (arxiv_id) arxiv_id, id, content
            FROM chunks WHERE chunk_index BETWEEN 3 AND 40
            ORDER BY arxiv_id, random()
        """)
        papers = cur.fetchall()
    conn.close()

    questions = load_existing()
    done = {q["arxiv_id"] for q in questions}
    pending = [p for p in papers if p[0] not in done]
    print(f"{len(done)} already done, {len(pending)} pending")

    for arxiv_id, chunk_id, content in pending:
        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
                messages=[{"role": "user", "content": PROMPT.format(text=content)}],
                temperature=0.3,
                max_tokens=800,
            )
        except RateLimitError:
            print(
                "\nRate limit reached. Progress is saved: run the script again later."
            )
            break
        question = (response.choices[0].message.content or "").strip()
        if question:
            questions.append(
                {"arxiv_id": arxiv_id, "chunk_id": chunk_id, "question": question}
            )
            save(questions)
            print(f"{arxiv_id}: {question}")
        time.sleep(1)


if __name__ == "__main__":
    main()
