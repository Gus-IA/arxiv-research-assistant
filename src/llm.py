import os
import re
from dotenv import load_dotenv
from groq import Groq

from retrieval import search_and_rerank

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about scientific papers. "
    "Answer strictly from the provided excerpts and do not add outside knowledge. "
    "Put the excerpt number in brackets right after EVERY claim, e.g. 'JEDI compresses "
    "demonstrations [2].' Never attribute a result to a paper unless that excerpt "
    "states it. If different excerpts come from different papers, keep them separate. "
    "Keep the answer under 200 words. If the excerpts do not contain the answer, "
    "say so clearly."
)

MIN_RERANK_SCORE = -5.0  # removes obvious noise; calibrate again in the evaluation


def cited_numbers(answer):
    """Return the excerpt numbers cited in the answer, e.g. [1], [2][3] or [1, 2]."""
    numbers = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        numbers.update(int(n) for n in re.findall(r"\d+", group))
    return numbers


def build_context(chunks):
    """Number each excerpt so the model can cite it."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c['title']} ({c['arxiv_id']})\n{c['content']}")
    return "\n\n".join(parts)


def ask(question, top_k=5, category=None, date_from=None):
    """Retrieve the best chunks, answer using only them, return only the cited sources."""
    chunks = search_and_rerank(
        question, top_k=top_k, category=category, date_from=date_from
    )
    chunks = [c for c in chunks if c["rerank_score"] >= MIN_RERANK_SCORE]
    if not chunks:
        return "No relevant excerpts were found for this question.", []

    # Same numbering as in the prompt, so [n] in the answer maps to chunk n
    for i, c in enumerate(chunks, start=1):
        c["n"] = i

    user_prompt = f"Excerpts:\n{build_context(chunks)}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    choice = response.choices[0]
    answer = choice.message.content
    if choice.finish_reason == "length":
        answer += "\n\n[Answer truncated: increase max_tokens]"

    cited = cited_numbers(answer)
    return answer, [c for c in chunks if c["n"] in cited]


if __name__ == "__main__":
    answer, sources = ask("How do robots learn manipulation from demostrations?")
    print(answer)
    print("\nSources:")
    for i, s in enumerate(sources, start=1):
        print(f"[{i}] {s['title']} ({s['arxiv_id']})")
