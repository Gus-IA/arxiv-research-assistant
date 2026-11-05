import os
import sys
from datetime import date, timedelta

import streamlit as st
from llm import ask, DEFAULT_METHOD
from retrieval import RETRIEVAL_METHODS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm import ask

CATEGORIES = {
    "All": None,
    "Robotics (cs.RO)": "cs.RO",
    "Machine Learning (cs.LG)": "cs.LG",
    "Language (cs.CL)": "cs.CL",
}


def group_sources(chunks):
    """Group chunks by paper, keeping the excerpt numbers used in the answer."""
    papers = {}
    for c in chunks:
        paper = papers.setdefault(c["arxiv_id"], {**c, "numbers": []})
        paper["numbers"].append(c["n"])
    return list(papers.values())


def render_sources(papers):
    for p in papers:
        numbers = ", ".join(f"[{n}]" for n in p["numbers"])
        st.markdown(f"**{numbers} {p['title']}**")
        st.caption(f"{p['authors'][:100]} · {p['published']} · {p['category']}")
        st.markdown(f"[Open PDF]({p['pdf_url']})")


st.set_page_config(page_title="arXiv Research Assistant", page_icon="📚", layout="wide")
st.title("📚 arXiv Research Assitant")
st.caption("Ask about recent papers. Answers cite the excerpts they come from.")

with st.sidebar:
    st.header("Filters")
    category_label = st.selectbox("Category", list(CATEGORIES))
    method = st.selectbox(
        "Retrieval method",
        RETRIEVAL_METHODS,
        index=RETRIEVAL_METHODS.index(DEFAULT_METHOD),
    )
    use_date = st.checkbox("Only papers since a date")
    date_from = (
        st.date_input("Published since", value=date.today() - timedelta(days=30))
        if use_date
        else None
    )
    top_k = st.slider("Excerpts used", 3, 10, 5)

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("papers"):
            with st.expander(f"Sources ({len(item['papers'])} papers)"):
                render_sources(item["papers"])

if question := st.chat_input("Ask about the papers..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            answer, chunks = ask(
                question,
                top_k=top_k,
                category=CATEGORIES[category_label],
                date_from=date_from,
                method=method,
            )
        papers = group_sources(chunks)
        st.markdown(answer)
        if papers:
            with st.expander(f"Sources ({len(papers)} papers)"):
                render_sources(papers)

    st.session_state.history.append(
        {"role": "assitant", "content": answer, "papers": papers}
    )
