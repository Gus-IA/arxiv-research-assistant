import os

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


def build_context(chunks):
    """Number each excerpt so the model can cite it."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] {c['title']} ({c['arxiv_id']})\n{c['content']}")
    return "\n\n".join(parts)


def ask(question, top_k=5, category=None, date_from=None):
    """Retrieve the best chunks, then ask the LLM to answer using only them."""
    chunks = search_and_rerank(
        question, top_k=top_k, category=category, date_from=date_from
    )
    if not chunks:
        return "No relevant excerpts were found for this question.", []

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
    # finish_reason == tokens máximos
    if choice.finish_reason == "length":
        answer += "\n\n[Answer truncated: increase max_tokens]"
    return answer, chunks


if __name__ == "__main__":
    answer, sources = ask("How do robots learn manipulation from demostrations?")
    print(answer)
    print("\nSources:")
    for i, s in enumerate(sources, start=1):
        print(f"[{i}] {s['title']} ({s['arxiv_id']})")
