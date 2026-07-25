from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import chromadb
import feedparser

from ..config import load_config


class HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text_parts: list[str] = []

    def handle_data(self, d: str) -> None:
        self.text_parts.append(d)

    def get_data(self) -> str:
        return "".join(self.text_parts)


def clean_html(raw_html: str) -> str:
    """
    Strips raw HTML tags (<p>, <a>, <br>, etc.) from feed text
    to prevent vector embedding contamination.
    """
    if not raw_html:
        return ""
    stripper = HTMLStripper()
    stripper.feed(raw_html)
    text = stripper.get_data()
    # Normalize multiple whitespace characters
    return re.sub(r"\s+", " ", text).strip()


def get_chroma_collection():
    cfg = load_config()
    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts", "artifacts"))
    chroma_path = artifacts_dir / "chroma"
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(name="market_news")


def ingest_rss_feed(feed_url: str) -> int:
    """
    Parses an RSS feed, cleans HTML tags, and persists text chunks to ChromaDB.
    """
    parsed = feedparser.parse(feed_url)
    collection = get_chroma_collection()

    documents = []
    metadatas = []
    ids = []

    for idx, entry in enumerate(parsed.entries):
        raw_content = entry.get("summary") or entry.get("description") or entry.get("title", "")
        cleaned_text = clean_html(raw_content)
        if not cleaned_text:
            continue

        doc_id = entry.get("id") or entry.get("link") or f"{feed_url}_{idx}"
        title = clean_html(entry.get("title", ""))
        published = entry.get("published", "")

        documents.append(f"{title}: {cleaned_text}")
        metadatas.append({"title": title, "published": published, "url": entry.get("link", "")})
        ids.append(str(doc_id))

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    return len(documents)


def query_news_chunks(query: str, n_results: int = 3) -> list[str]:
    """
    Queries ChromaDB for top relevant news chunks matching the query.
    """
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        return [str(d) for d in docs if d]
    return []
