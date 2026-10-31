import os
import re

import arxiv
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import requests

load_dotenv()

# --- Configuration ---
CATEGORIES = ["cs.RO", "cs.LG", "cs.CL"]  # robotics, machine learning, NLP
PAPERS_PER_CATEGORY = 20  # start small, increase later
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200
PAPERS_DIR = "data/papers"

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POTSGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5433),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def clean_text(text):
    """Remove NUL bytes (PostgreSQL rejects them) and collapse whitespace."""
    text = text.replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


def extract_text(pdf_path):
    """Extract the text of a PDF. Returns an empty string if it fails"""
    try:
        reader = PdfReader(pdf_path)
        return clean_text(" ".join(page.extract_text() or "" for page in reader.pages))
    except Exception as e:
        print(f" Could not read {pdf_path}: {e}")
        return ""


def split_into_chunks(text):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE].strip()
        if len(piece) > 100:
            chunks.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def pdf_url_for(result):
    """Build the PDF URL from the arXiv id (works with any version of the arxiv library)."""
    return f"https://arxiv.org/pdf/{result.get_short_id()}"


def download_pdf(result, dirpath):
    """Download a paper PDF with requests and return its local path."""
    arxiv_id = result.get_short_id()
    path = os.path.join(dirpath, f"{arxiv_id.replace('/', '_')}.pdf")
    if os.path.exists(path):
        return path
    response = requests.get(
        pdf_url_for(result),
        headers={"User-Agent": "arxiv-research-assistant (personal learning project)"},
        timeout=60,
    )
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    time.sleep(3)  # be polite with arXiv servers
    return path


def paper_exists(cur, arxiv_id):
    cur.execute("SELECT 1 FROM papers WHERE arxiv_id = %s", (arxiv_id,))
    return cur.fetchone() is not None


def ingest_paper(conn, result, pdf_path):
    arxiv_id = result.get_short_id()
    text = extract_text(pdf_path)
    chunks = split_into_chunks(text)
    if not chunks:
        print(f"  No text extracted for {arxiv_id}, skipping")
        return False

    # Embedding: text -> vector (one batch per paper)
    embeddings = model.encode(chunks)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO papers (arxiv_id, title, authors, abstract, category, published, pdf_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (arxiv_id) DO NOTHING
            """,
            (
                arxiv_id,
                clean_text(result.title),
                ", ".join(a.name for a in result.authors),
                clean_text(result.summary),
                result.primary_category,
                result.published.date(),
                pdf_url_for(result),
            ),
        )
        rows = [
            (arxiv_id, i, chunk, "[" + ",".join(map(str, emb.tolist())) + "]")
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        execute_values(
            cur,
            "INSERT INTO chunks (arxiv_id, chunk_index, content, embedding) VALUES %s",
            rows,
            template="(%s, %s, %s, %s::vector)",
        )
    conn.commit()
    return True


def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    conn = get_conn()
    client = arxiv.Client(page_size=PAPERS_PER_CATEGORY, delay_seconds=3, num_retries=3)

    total = 0
    for category in CATEGORIES:
        print(f"\nCategory {category}")
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=PAPERS_PER_CATEGORY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        for result in tqdm(client.results(search), total=PAPERS_PER_CATEGORY):
            arxiv_id = result.get_short_id()

            # Skip papers already indexed so the script can be re-run safely
            with conn.cursor() as cur:
                if paper_exists(cur, arxiv_id):
                    continue

            try:
                pdf_path = download_pdf(result, PAPERS_DIR)
                if ingest_paper(conn, result, pdf_path):
                    total += 1
            except Exception as e:
                conn.rollback()
                print(f"  Error with {arxiv_id}: {e}")

    conn.close()
    print(f"\nIngestion finished: {total} new papers indexed")


if __name__ == "__main__":
    main()
