import os
import re
import json
import asyncio
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import httpx
from bs4 import BeautifulSoup

try:
    import PyPDF2  # noqa: WPS433
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyPDF2 is required for PDF extraction. Install it via `pip install PyPDF2`."
    ) from exc

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------

# NOTE: *Do NOT* hard-code a cruise-line name here. Everything is derived from
# the JSON input structure.

JSON_PATH = "cruise_docs.json"  # Default path – can be overridden via CLI

# Ensure we have an OpenAI key *somewhere* before we do any remote calls. You
# may prefer to export it from the shell rather than keeping a hard-coded key.
if "OPENAI_API_KEY" not in os.environ:
 
# Initialise logger early so downstream LightRAG components inherit settings
setup_logger("lightrag", level="INFO")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Return a filesystem-friendly, lowercase slug for *text*."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def build_metadata_header(name: str, url: str, doc_type: str) -> str:
    """Return a formatted metadata block for *name* and *url*."""

    return (
        f"Title: {name}\n"
        f"Source URL: {url}\n"
        f"Document Type: {doc_type}\n"
        f"Date Processed: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"---\n"
    )

# ---------------------------------------------------------------------------
# Remote content fetching helpers (mostly copied from the Carnival script)
# ---------------------------------------------------------------------------

async def fetch_html_text(url: str, name: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """Download *url* HTML and return cleaned text plus any linked PDF docs.

    Returns ``(html_text, list_of_(pdf_text,url))``.
    """
    linked_pdf_docs: List[Tuple[str, str]] = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

        # Extract linked PDFs (absolute or relative)
        pdf_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                pdf_links.add(href)

        # Resolve relative URLs using urljoin
        from urllib.parse import urljoin, urlparse

        pdf_links_resolved = {urljoin(url, link) for link in pdf_links}

        # ------------------------------------------------------------------
        # Domain filtering: keep only PDFs hosted on the SAME (sub)domain
        # as the parent HTML. This avoids pulling in documents from other
        # cruise-line websites (e.g. Carnival PDFs appearing in Princess graphs).
        # ------------------------------------------------------------------
        origin_host = urlparse(url).hostname or ""
        origin_host_stripped = origin_host.lstrip("www.")
        pdf_links_resolved = {
            plink for plink in pdf_links_resolved
            if (
                (h := urlparse(plink).hostname) and
                (h.lstrip("www.").endswith(origin_host_stripped))
            )
        }

        # Fetch PDFs concurrently
        if pdf_links_resolved:
            pdf_texts = await asyncio.gather(
                *[fetch_pdf_text(plink, f"{name} - Linked PDF") for plink in pdf_links_resolved]
            )
            linked_pdf_docs = [
                (t, plink) for t, plink in zip(pdf_texts, pdf_links_resolved) if t
            ]

        # Prepare HTML text: strip scripts/styles & condense whitespace
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        full_text = build_metadata_header(name, url, "HTML Document") + text.strip()
        return (full_text if len(full_text) >= 50 else None), linked_pdf_docs
    except Exception as exc:  # noqa: WPS420
        print(f"[HTML ERROR] {url}: {exc}")
        return None, linked_pdf_docs


async def fetch_pdf_text(url: str, name: str) -> Optional[str]:
    """Download remote PDF at *url* and extract its text."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        reader = PyPDF2.PdfReader(BytesIO(resp.content))
        text_parts: List[str] = [build_metadata_header(name, url, "PDF Document")]
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                cleaned = page_text.strip().replace("\n\n", "\n").replace("  ", " ")
                text_parts.append(cleaned)
        full_text = "\n".join(text_parts)
        return full_text if len(full_text) > 50 else None
    except Exception as exc:  # noqa: WPS420
        print(f"[PDF ERROR] {url}: {exc}")
        return None


async def process_document(doc: Dict[str, str]) -> List[Tuple[Optional[str], str]]:
    """Return a list of ``(extracted_text, source_url)`` pairs for *doc*."""
    url = doc["url"]
    name = doc["name"]
    file_type = doc["file_type"]

    results: List[Tuple[Optional[str], str]] = []

    if file_type == "html":
        html_text, linked_pdfs = await fetch_html_text(url, name)
        results.append((html_text, url))
        results.extend(linked_pdfs)
    elif file_type == "pdf":
        pdf_text = await fetch_pdf_text(url, name)
        results.append((pdf_text, url))
    else:
        print(f"[SKIP] Unsupported file type '{file_type}' for {url}")

    return results

# ---------------------------------------------------------------------------
# LightRAG initialisation helper
# ---------------------------------------------------------------------------

async def initialise_rag(working_dir: str, workspace: str | None = None) -> LightRAG:
    """Return a configured LightRAG instance bound to *working_dir*.

    A dedicated *workspace* name is supplied so that even if multiple LightRAG
    instances share the same external storage backend (e.g. Redis, PostgreSQL),
    their data remains logically isolated.  When *workspace* is ``None`` the
    default LightRAG behaviour (environment‐variable or empty string) is used.
    """
    os.makedirs(working_dir, exist_ok=True)
    rag = LightRAG(
        working_dir=working_dir,
        workspace=workspace or "",  # Fallback to default behaviour if not provided
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

# Expected JSON schema:
# [
#   {
#     "Cruise_Line_Name": {
#       "Document_Type": "URL",
#       ...
#     }
#   },
#   ...
# ]
# Example:
#   {
#     "Carnival_Cruise_Line": {
#       "Ticket_Contract": "https://...",
#       "FAQ": "https://..."
#     }
#   }

def detect_file_type_from_url(url: str) -> str:
    """Return 'pdf' or 'html' based on *url* extension (default to 'html')."""
    if url.lower().endswith(".pdf"):
        return "pdf"
    return "html"


def normalize_doc_type(doc_type: str) -> str:
    """Convert doc type keys like 'Ticket_Contract' to 'Ticket Contract'."""
    return doc_type.replace("_", " ").strip().title()


def parse_docs_json(json_path: str) -> Dict[str, List[Dict[str, str]]]:
    """Return mapping of ``cruise_line -> list[doc_meta]`` from *json_path*.

    The expected JSON structure is a *list* where each element is an *object*
    whose single key is the cruise-line name and whose value is an *object*
    mapping *document type* → *URL*.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    docs_by_line: Dict[str, List[Dict[str, str]]] = {}
    if not isinstance(data, list):
        raise ValueError("Top-level JSON structure must be a list.")

    for entry in data:
        if not isinstance(entry, dict):
            print(f"[WARN] Skipping unexpected entry type: {entry!r}")
            continue
        for cruise_line, docs in entry.items():
            if not isinstance(docs, dict):
                print(f"[WARN] Expected an object for cruise-line '{cruise_line}', got {type(docs)}")
                continue
            processed_docs: List[Dict[str, str]] = []
            for doc_type, url in docs.items():
                file_type = detect_file_type_from_url(url)
                processed_docs.append({
                    "name": normalize_doc_type(doc_type),
                    "file_type": file_type,
                    "url": url,
                })
            if not processed_docs:
                print(f"[WARN] No documents found for cruise line '{cruise_line}'.")
            docs_by_line[cruise_line] = processed_docs
    return docs_by_line

# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------

async def ingest_all_cruise_docs(json_path: str, *, test_queries: Optional[List[str]] | None = None) -> None:
    """Ingest documents for *all* cruise lines defined in *json_path*."""

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. "
            "Please set this variable before running."
        )
        return

    docs_by_line = parse_docs_json(json_path)
    if not docs_by_line:
        print("No cruise lines found in JSON – aborting.")
        return

    for cruise_line, docs_meta in docs_by_line.items():
        print(f"\n=== Processing '{cruise_line}' ({len(docs_meta)} docs) ===")
        # Extract & clean documents -------------------------------------------------
        nested_results = await asyncio.gather(*[process_document(d) for d in docs_meta])

        documents: List[str] = []
        sources: List[str] = []
        for doc_results in nested_results:
            for text, url in doc_results:
                if text:
                    documents.append(text)
                    sources.append(url)
                else:
                    print(f"[SKIP] No content extracted from {url}")

        if not documents:
            print(f"No valid documents extracted for '{cruise_line}' – skipping.")
            continue

        # Create a dedicated graph per cruise line ----------------------------------
        working_dir = f"./{slugify(cruise_line)}_graph"
        workspace = slugify(cruise_line)
        rag = await initialise_rag(working_dir, workspace)
        await rag.ainsert(documents, file_paths=sources)  # Use URLs as file_path meta
        print(f"Ingestion complete for '{cruise_line}'.")

        # Optional: run sample queries ---------------------------------------------
        if test_queries:
            print("\n--- SAMPLE QUERIES ---")
            for q in test_queries:
                print(f"\nQuery: {q}")
                try:
                    result = await rag.aquery(q, param=QueryParam(mode="hybrid"))
                    print(result)
                except Exception as exc:  # noqa: WPS420
                    print(f"Error running query: {exc}")

        # Summary ------------------------------------------------------------------
        status_path = os.path.join(working_dir, "kv_store_doc_status.json")
        if os.path.exists(status_path):
            with open(status_path, "r", encoding="utf-8") as fp:
                info = json.load(fp)
            print(f"Total documents indexed for '{cruise_line}': {len(info)}")

        # Gracefully close internal LightRAG storages
        await rag.finalize_storages()

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

DEFAULT_TEST_QUERIES = [
    "What is the cancellation policy?",
    "What accessibility accommodations are available for wheelchair users?",
    "What are the safety protocols on this cruise line?",
    "What are the terms and conditions for shore excursions?",
]

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Ingest cruise-line documents into LightRAG graphs.")
    parser.add_argument("--json", default=JSON_PATH, help="Path to the JSON file describing documents.")
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Skip running the default sample queries after ingestion.",
    )
    args = parser.parse_args()

    asyncio.run(
        ingest_all_cruise_docs(
            args.json,
            test_queries=None if args.skip_queries else DEFAULT_TEST_QUERIES,
        )
    ) 