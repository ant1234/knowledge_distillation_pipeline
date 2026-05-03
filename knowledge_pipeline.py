"""
Knowledge Distillation Pipeline
=================================
Crawls any web library, extracts PDFs, chunks and embeds text,
deduplicates via Qdrant vector search, distils novel claims using a
local LLM, and exports a structured knowledge base to CSV.

Set LIBRARY_URL in .env to point at any library index page.

Usage:
  python knowledge_pipeline.py crawl                # Stage 1: build link manifest
  python knowledge_pipeline.py run                  # Stages 2-4: process all PDFs
  python knowledge_pipeline.py run --limit 3        # Process at most N unprocessed
  python knowledge_pipeline.py single /path/to.pdf  # Test with a local PDF
  python knowledge_pipeline.py export               # Stage 5: export claims to CSV
  python knowledge_pipeline.py purge                # Wipe DB and progress
  python knowledge_pipeline.py status               # Show progress summary
"""

import os
import re
import csv
import json
import time
import argparse
import hashlib
import logging
import threading
import io as _io
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
import pandas as pd

import pymupdf as fitz
_LAYOUT_AVAILABLE = hasattr(fitz, '_get_layout')

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue,
)

# ─────────────────────────────────────────────
# LOAD .env BEFORE reading any os.getenv
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LIBRARY_URL          = os.getenv("LIBRARY_URL",        "")
CRAWL_DELAY          = float(os.getenv("CRAWL_DELAY",  "3"))
REQUEST_TIMEOUT      = int(os.getenv("REQUEST_TIMEOUT","30"))

CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE",     "400"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP",  "60"))
MAX_CHUNK_CHARS      = int(os.getenv("MAX_CHUNK_CHARS","3500"))

SIMILARITY_DISCARD   = float(os.getenv("SIMILARITY_DISCARD","0.85"))
SIMILARITY_CHECK     = float(os.getenv("SIMILARITY_CHECK",  "0.70"))

DISTILL_MAX_CLAIMS   = int(os.getenv("DISTILL_MAX_CLAIMS",  "20"))
TEMPERATURE          = float(os.getenv("TEMPERATURE",       "0.3"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS",          "4096"))
MODEL_NAME           = os.getenv("MODEL_NAME",      "deepseek-r1:14b")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PAGE_RATIO_TOLERANCE = float(os.getenv("PAGE_RATIO_TOLERANCE","0.03"))
MIN_IMAGE_PIXELS     = int(os.getenv("MIN_IMAGE_PIXELS",     "10000"))

QDRANT_URL           = os.getenv("QDRANT_URL",        "")
QDRANT_COLLECTION    = os.getenv("QDRANT_COLLECTION", "knowledge_chunks")

DATA_DIR             = Path(os.getenv("DATA_DIR", "./pipeline_data"))
IMAGES_DIR           = DATA_DIR / "images"
QDRANT_PATH          = DATA_DIR / "qdrant_db"
MANIFEST_FILE        = DATA_DIR / "manifest.json"
PROGRESS_FILE        = DATA_DIR / "progress.json"
PROCESSED_TITLES     = DATA_DIR / "processed_titles.json"
IDEAS_CSV            = DATA_DIR / "ideas.csv"
LOG_FILE             = DATA_DIR / "pipeline.log"

# nomic-embed-text produces 768-dimensional vectors
EMBEDDING_DIM        = 768

# Standard page aspect ratios used to detect full-page background images
_PAGE_RATIOS = [0.7071, 1.4142, 0.7727, 1.2941, 0.8165, 1.2247]

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

if _LAYOUT_AVAILABLE:
    log.info("pymupdf_layout detected - enhanced page layout analysis enabled")
else:
    log.info("pymupdf_layout not installed - using standard text extraction")

# ─────────────────────────────────────────────
# LLM + EMBEDDINGS
# ─────────────────────────────────────────────

def get_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        timeout=120,
    )

def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

def embed_text(text: str) -> list[float]:
    # Truncate by words first, then chars — handles token-dense content
    # (barcodes, hex strings, codes) where chars-per-token ratio is high
    words = text.split()
    if len(words) > CHUNK_SIZE:
        text = " ".join(words[:CHUNK_SIZE])
    if len(text) > MAX_CHUNK_CHARS:
        text = text[:MAX_CHUNK_CHARS]
    return get_embeddings().embed_query(text)

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _invoke_with_timeout(llm, prompt, timeout_seconds=120):
    """Run llm.invoke in a thread — kill it if it exceeds timeout."""
    result = [None]
    error  = [None]

    def target():
        try:
            result[0] = llm.invoke(prompt).content
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        log.warning(f"  LLM call timed out after {timeout_seconds}s — skipping")
        return None  # thread keeps running as daemon but we move on

    if error[0]:
        raise error[0]

    return result[0]

# ─────────────────────────────────────────────
# QDRANT
# ─────────────────────────────────────────────

def get_qdrant() -> QdrantClient:
    """
    Return a Qdrant client.
    Blank QDRANT_URL -> local in-process storage at QDRANT_PATH.
    Set QDRANT_URL to connect to a running Qdrant server.
    """
    if QDRANT_URL:
        log.info(f"Connecting to Qdrant server: {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        log.info(f"Created collection '{QDRANT_COLLECTION}'")
    else:
        log.info(f"Collection '{QDRANT_COLLECTION}' ready")


def collection_count(client: QdrantClient) -> int:
    try:
        return client.count(collection_name=QDRANT_COLLECTION).count
    except Exception:
        return 0

# ─────────────────────────────────────────────
# PERSISTENCE HELPERS
# ─────────────────────────────────────────────

def load_json(path: Path, default):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────
# STAGE 1 - CRAWL
# ─────────────────────────────────────────────

def crawl_library() -> list[dict]:
    """
    Crawl LIBRARY_URL, collect every PDF link and its metadata.
    Writes / merges results into manifest.json.
    Safe to re-run - merges new entries without overwriting existing ones.
    """
    if not LIBRARY_URL:
        raise ValueError("LIBRARY_URL is not set. Add it to your .env file.")

    log.info(f"Crawling: {LIBRARY_URL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "KnowledgePipelineBot/1.0 (academic research)"})

    resp = session.get(LIBRARY_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Derive the base domain for relative-URL resolution
    from urllib.parse import urlparse
    parsed   = urlparse(LIBRARY_URL)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    # Collect all internal document links
    doc_links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = href if href.startswith("http") else base_url + href
        # Accept direct PDF links from any path on the same domain
        if full.lower().endswith(".pdf"):
            if full not in doc_links:
                doc_links.append(full)
            continue
        # Also follow library section pages that may contain more PDF links
        if "/library/" in full and full != LIBRARY_URL and full not in doc_links:
            doc_links.append(full)

    log.info(f"Found {len(doc_links)} candidate document pages")

    # Load existing manifest to allow safe re-runs
    existing  = load_json(MANIFEST_FILE, [])
    seen_pdfs = {e["pdf_url"] for e in existing}
    manifest  = list(existing)

    for idx, doc_url in enumerate(doc_links):
        try:
            log.info(f"[{idx+1}/{len(doc_links)}] {doc_url}")

            # If the link is itself a PDF, use it directly without fetching HTML
            if doc_url.lower().endswith(".pdf"):
                pdf_url = doc_url
                if pdf_url in seen_pdfs:
                    continue
                seen_pdfs.add(pdf_url)

                title  = Path(pdf_url).stem.replace("_", " ").replace("-", " ").title()
                author = "Unknown"
                year   = _extract_year_from_url(pdf_url)
                dtype  = "publication"

                entry = {
                    "id":       hashlib.md5(pdf_url.encode()).hexdigest()[:10],
                    "title":    title.strip(),
                    "author":   author,
                    "year":     year,
                    "type":     dtype,
                    "pdf_url":  pdf_url,
                    "page_url": pdf_url,
                    "processed": False,
                }
                manifest.append(entry)
                save_json(MANIFEST_FILE, manifest)
                log.info(f"  + {title[:70]} [{dtype}]")
                time.sleep(0.5)   # shorter delay - no HTML fetch needed
                continue

            # Otherwise fetch the HTML page and look for a PDF link within it
            time.sleep(CRAWL_DELAY)
            r = session.get(doc_url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                log.warning(f"  Skipped (HTTP {r.status_code})")
                continue

            dsoup = BeautifulSoup(r.text, "html.parser")

            pdf_url = None
            for a in dsoup.find_all("a", href=True):
                if a["href"].lower().endswith(".pdf"):
                    pdf_url = (
                        a["href"] if a["href"].startswith("http")
                        else base_url + a["href"]
                    )
                    break

            if not pdf_url or pdf_url in seen_pdfs:
                continue
            seen_pdfs.add(pdf_url)

            title  = _meta_or_text(dsoup, ["h1", ".title", ".object-title"]) or Path(pdf_url).stem
            author = _meta_or_text(dsoup, [".author", ".creator", "[itemprop='author']"]) or "Unknown"
            year   = _extract_year(dsoup)
            dtype  = _infer_doc_type(title, dsoup)

            entry = {
                "id":       hashlib.md5(pdf_url.encode()).hexdigest()[:10],
                "title":    title.strip(),
                "author":   author.strip(),
                "year":     year,
                "type":     dtype,
                "pdf_url":  pdf_url,
                "page_url": doc_url,
                "processed": False,
            }
            manifest.append(entry)
            save_json(MANIFEST_FILE, manifest)
            log.info(f"  + {title[:70]} ({year}) [{dtype}]")

        except Exception as exc:
            log.error(f"  Error on {doc_url}: {exc}")

    save_json(MANIFEST_FILE, manifest)
    log.info(f"Manifest saved: {len(manifest)} PDFs -> {MANIFEST_FILE}")
    return manifest


def _meta_or_text(soup, selectors: list) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el.get_text(separator=" ", strip=True)
    for prop in ["og:title", "DC.title", "citation_title"]:
        m = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if m and m.get("content"):
            return m["content"]
    return ""

def _extract_year(soup) -> str:
    m = re.search(r"\b(1[0-9]\d{2}|20[012]\d)\b", soup.get_text())
    return m.group(0) if m else "Unknown"

def _extract_year_from_url(url: str) -> str:
    """Extract a 4-digit year from a PDF filename if present."""
    m = re.search(r'(1[89]\d{2}|20[012]\d)', url)
    return m.group(0) if m else "Unknown"

def _infer_doc_type(title: str, soup) -> str:
    t = title.lower()
    p = soup.get_text().lower()
    if any(w in t for w in ["report", "excavation", "season", "field"]):
        return "field_report"
    if any(w in t for w in ["journal", "article", "notes", "bulletin"]):
        return "article"
    if "dissertation" in t or "thesis" in t:
        return "thesis"
    if any(w in p for w in ["volume", "chapter", "isbn"]):
        return "book"
    return "publication"

# ─────────────────────────────────────────────
# STAGE 2 - EXTRACT
# ─────────────────────────────────────────────

def _is_page_background(width: int, height: int) -> bool:
    """
    True if this image matches a standard page aspect ratio AND is large -
    indicating a full-page background scan rather than a content illustration.
    """
    if width == 0 or height == 0:
        return False
    ratio = width / height
    for pr in _PAGE_RATIOS:
        if abs(ratio - pr) < PAGE_RATIO_TOLERANCE and width * height > 500_000:
            return True
    return False


def _extract_caption_near_image(page, img_xref: int) -> str:
    """
    Find a figure caption near an image on the page.
    Uses get_image_info(xrefs=True) for accurate bbox.
    Falls back to width/height matching for JPXDecode images where
    get_image_bbox() raises 'bad image name'.
    Searches within 200 pts from any edge of the image rect.
    """
    try:
        img_rect: Optional[fitz.Rect] = None

        # Method 1: xref match - requires xrefs=True to populate the xref field
        for item in page.get_image_info(xrefs=True):
            if item.get("xref") == img_xref:
                bbox = item.get("bbox")
                if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    img_rect = fitz.Rect(bbox)
                break

        # Method 2: size-based fallback for JPXDecode / Form XObject images
        if img_rect is None:
            size_map = {img[0]: (img[2], img[3]) for img in page.get_images(full=True)}
            target   = size_map.get(img_xref)
            if target:
                for item in page.get_image_info(xrefs=True):
                    if item.get("width") == target[0] and item.get("height") == target[1]:
                        bbox = item.get("bbox")
                        if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            img_rect = fitz.Rect(bbox)
                        break

        if img_rect is None or img_rect.is_empty:
            return ""

        caption_re = re.compile(
            r"\b(fig\.?|figure|plate|pl\.?|photo|photograph|map|plan|table)\b",
            re.IGNORECASE,
        )
        best_text = ""
        best_dist = float("inf")

        for block in page.get_text("blocks"):
            bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
            text = block[4].strip()
            if not text or not caption_re.search(text):
                continue
            # Euclidean distance from nearest edge of image to nearest edge of block
            dx   = max(0.0, max(bx0 - img_rect.x1, img_rect.x0 - bx1))
            dy   = max(0.0, max(by0 - img_rect.y1, img_rect.y0 - by1))
            dist = (dx**2 + dy**2) ** 0.5
            if dist < 200 and dist < best_dist:
                best_dist = dist
                best_text = text

        return best_text[:400] if best_text else ""
    except Exception:
        return ""


def extract_pdf(pdf_path: Path, doc_meta: dict) -> tuple[list[str], list[dict]]:
    """
    Extract text chunks, content images (filtered), and tables from a PDF.
    Returns (chunks, image_records).
    """
    doc_id   = doc_meta.get("id", hashlib.md5(str(pdf_path).encode()).hexdigest()[:10])
    doc_name = _safe_name(doc_meta.get("title", pdf_path.stem))
    img_dir  = IMAGES_DIR / doc_name
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    full_text_pages: list[str] = []
    image_records:   list[dict] = []
    bg_discarded = tiny_discarded = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ── Text - use layout engine if available for better reading order ──
        if _LAYOUT_AVAILABLE:
            page_text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE,
                sort=True,
            )
        else:
            page_text = page.get_text("text")

        # ── Tables ──
        try:
            for t_idx, table in enumerate(page.find_tables()):
                df = pd.DataFrame(table.extract())
                page_text += (
                    f"\n[TABLE p.{page_num+1} t.{t_idx+1}]\n"
                    + df.to_string(index=False, header=False)
                    + "\n"
                )
        except Exception:
            pass

        # ── Images ──
        seen_xrefs: set[int] = set()
        content_img_count    = 0

        for img_info in page.get_images(full=True):
            try:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                base_img  = doc.extract_image(xref)
                img_bytes = base_img["image"]
                img_w     = base_img.get("width",  0)
                img_h     = base_img.get("height", 0)

                # Discard tiny decorative images
                if img_w * img_h < MIN_IMAGE_PIXELS:
                    tiny_discarded += 1
                    continue

                # Discard full-page background scans
                if _is_page_background(img_w, img_h):
                    bg_discarded += 1
                    continue

                content_img_count += 1
                img_path = img_dir / f"page_{page_num+1:04d}_img_{content_img_count:03d}.png"
                Image.open(_io.BytesIO(img_bytes)).convert("RGB").save(img_path, "PNG")

                caption = _extract_caption_near_image(page, xref)

                image_records.append({
                    "doc_id":    doc_id,
                    "doc_title": doc_meta.get("title", ""),
                    "page":      page_num + 1,
                    "img_index": content_img_count,
                    "path":      str(img_path),
                    "caption":   caption,
                })

                if caption:
                    page_text += f"\n[FIGURE CAPTION p.{page_num+1}]: {caption}\n"

            except Exception as exc:
                log.debug(f"  Image error p{page_num+1}: {exc}")

        full_text_pages.append(page_text)

    doc.close()

    if bg_discarded or tiny_discarded:
        log.info(f"  Discarded: {bg_discarded} backgrounds, {tiny_discarded} tiny images")

    chunks = _chunk_text("\n".join(full_text_pages))
    log.info(
        f"  Extracted {len(chunks)} chunks, {len(image_records)} content images "
        f"from {pdf_path.name}"
    )
    return chunks, image_records


def _chunk_text(text: str) -> list[str]:
    words   = text.split()
    step    = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks  = []
    trimmed = 0
    for i in range(0, max(1, len(words) - CHUNK_OVERLAP), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) < 50:
            continue
        if len(chunk) > MAX_CHUNK_CHARS:
            chunk = chunk[:MAX_CHUNK_CHARS]
            trimmed += 1
        chunks.append(chunk)
    if trimmed:
        log.warning(f"  {trimmed} chunks trimmed to {MAX_CHUNK_CHARS} chars")
    return chunks

def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-_]", "_", name)[:60]

# ─────────────────────────────────────────────
# STAGE 3 - DEDUPLICATE + EMBED (Qdrant)
# ─────────────────────────────────────────────

def embed_chunks_with_deduplication(
    chunks:     list[str],
    doc_meta:   dict,
    image_refs: list[dict],
    client:     QdrantClient,
    llm:        ChatOllama,
) -> tuple[int, int, int]:
    added = skipped = llm_checks = 0
    MAX_LLM_CHECKS_PER_DOC = 20

    # Truncate all chunks before batch embedding
    safe_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) > CHUNK_SIZE:
            chunk = " ".join(words[:CHUNK_SIZE])
        if len(chunk) > MAX_CHUNK_CHARS:
            chunk = chunk[:MAX_CHUNK_CHARS]
        safe_chunks.append(chunk)

    # Single batch call instead of one call per chunk
    try:
        vectors = get_embeddings().embed_documents(safe_chunks)
    except Exception as e:
        log.warning(f"  Batch embed failed ({e}), falling back to individual embedding")
        vectors = []
        for chunk in safe_chunks:
            try:
                vectors.append(embed_text(chunk))
            except Exception as e2:
                log.warning(f"  Skipping chunk (embed failed: {e2}): {chunk[:60]}...")
                vectors.append(None)

    is_empty = collection_count(client) == 0
    points: list[PointStruct] = []

    for chunk, vector in zip(safe_chunks, vectors):
        if vector is None:
            skipped += 1
            continue

        if is_empty:
            points.append(_make_point(chunk, vector, doc_meta))
            added += 1
            continue

        try:
            results = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                limit=1,
            )
            hits = results.points
        except Exception:
            points.append(_make_point(chunk, vector, doc_meta))
            added += 1
            continue

        if not hits:
            points.append(_make_point(chunk, vector, doc_meta))
            added += 1
            continue

        similarity = hits[0].score

        if similarity >= SIMILARITY_DISCARD:
            skipped += 1
            log.debug(f"  SKIP (sim={similarity:.3f}): {chunk[:80]}...")

        elif similarity >= SIMILARITY_CHECK:
            if llm_checks >= MAX_LLM_CHECKS_PER_DOC:
                # Cap reached — treat as novel rather than risk a hung call
                points.append(_make_point(chunk, vector, doc_meta))
                added += 1
                log.debug(f"  LLM cap reached, adding as novel: {chunk[:80]}...")
            else:
                llm_checks += 1
                best_chunk = hits[0].payload.get("text", "")
                if _llm_duplicate_check(chunk, best_chunk, llm):
                    skipped += 1
                    log.debug(f"  LLM-SKIP (sim={similarity:.3f}): {chunk[:80]}...")
                else:
                    points.append(_make_point(chunk, vector, doc_meta))
                    added += 1

        else:
            points.append(_make_point(chunk, vector, doc_meta))
            added += 1

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    log.info(f"  Chunks: {added} added, {skipped} skipped, {llm_checks} LLM checks")
    return added, skipped, llm_checks

def _make_point(chunk: str, vector: list[float], meta: dict) -> PointStruct:
    uid = hashlib.md5((chunk + meta.get("id", "")).encode()).hexdigest()
    return PointStruct(
        id=int(uid[:8], 16),
        vector=vector,
        payload={
            "text":   chunk,
            "doc_id": meta.get("id",     ""),
            "title":  meta.get("title",  ""),
            "author": meta.get("author", ""),
            "year":   meta.get("year",   ""),
            "type":   meta.get("type",   ""),
        },
    )


def _llm_duplicate_check(chunk_a: str, chunk_b: str, llm: ChatOllama) -> bool:
    prompt = (
        "Are these two passages conveying the same factual information?\n"
        "Answer with YES or NO only.\n\n"
        f"Passage A:\n{chunk_a[:600]}\n\nPassage B:\n{chunk_b[:600]}\n\nAnswer:"
    )
    response = _invoke_with_timeout(llm, prompt, timeout_seconds=30)
    if response is None:
        return False  # on timeout, assume not duplicate and continue
    answer = strip_think_tags(response).strip().upper()
    return answer.startswith("Y")

# ─────────────────────────────────────────────
# STAGE 4 - DISTIL
# ─────────────────────────────────────────────

DISTIL_PROMPT = """\
You are a research analyst who specialises in extracting precise, citable knowledge \
from academic and specialist publications.

Source document:
  Title:  {title}
  Author: {author}
  Year:   {year}
  Type:   {doc_type}

TEXT:
{text}

Task: Identify up to {max_claims} distinct factual claims, measurements, or \
interpretations that THIS document specifically contributes. You MUST name the \
specific subject by name in EVERY sentence of EVERY claim - never write "the tomb", \
"the structure", "the researchers", "this finding", or any other vague reference \
without the specific name. If the subject is Kay's tomb at Giza, write "Kay's tomb \
at Giza" every time, not "the tomb".

CRITICAL GROUNDING RULES - these override everything else:
- ONLY extract claims that are EXPLICITLY stated in the TEXT above.
- Do NOT use your training knowledge. Do NOT infer. Do NOT guess.
- If the TEXT does not clearly state a fact, omit it entirely. It is better to \
  return fewer claims than to invent or embellish.
- Every measurement, date, name, and location in your output MUST appear verbatim \
  in the TEXT above. If you cannot find it in the TEXT, do not write it.

Formatting rules:
- Each claim MUST be exactly 2-4 complete sentences of continuous flowing prose.
- NO bullet points, dashes, asterisks, or sub-items of any kind within a claim.
- Every sentence must name the specific subject (e.g. "Kay's tomb at Giza", \
  "the north wall of Kay's tomb", "Petrie's 1883 survey of the Great Pyramid").
- Sentence 1: name the specific subject AND state the finding with numbers, \
  measurements, or dates drawn directly from the TEXT.
- Sentence 2: provide further context about that same named subject as stated \
  in the TEXT - which wall, layer, material, or location.
- Sentence 3 (if relevant): method used or significance, as stated in the TEXT.
- If text contains [FIGURE CAPTION] markers, incorporate what the figure shows \
  into the relevant claim, naming the subject.
- EXCLUDE claims that only cite or summarise prior work without adding new data.
- EXCLUDE vague statements with no specific data, numbers, or named subjects.
- EXCLUDE any claim shorter than 2 full sentences.
- Start each claim with its number followed by a period, e.g. "1. Kay's tomb at \
  Giza measured..."
- Each numbered claim must be on its own line, not grouped into paragraphs.

Return ONLY a numbered list where each item is 2-4 sentences of flowing prose. \
No bullets. No dashes. No sub-items. No preamble. No commentary after the list.
"""

SUMMARY_PROMPT = """\
You are a research analyst extracting structured knowledge from academic publications.

Source: {title} - {author} ({year})

Top claims from this document:
{claims_text}

CRITICAL: Your title and summary must be based ONLY on the claims listed above. \
Do not introduce any facts, names, or details that do not appear in those claims.

Tasks:
1. Write a SHORT descriptive title (8 words maximum) that names the specific \
subject - the exact tomb, site, person, or object - and what the document \
contributes about it. Never use vague phrases like "conservation methods" \
without naming what was conserved and where.
2. Write a 2-3 sentence summary that names the specific subject in the first \
sentence. Every fact in the summary must come directly from the claims above.

Respond in this exact format and nothing else:
TITLE: <title>
SUMMARY: <summary>
"""

def distil_document(chunks: list[str], doc_meta: dict, llm: ChatOllama) -> list[dict]:
    """
    Distil chunks into labelled claims with a document-level summary.
    Returns a list of claim dicts.
    """
    sample = " ".join(" ".join(chunks).split()[:8000])

    try:
        response = _invoke_with_timeout(
            llm,
            DISTIL_PROMPT.format(
                title      = doc_meta.get("title",  "Unknown"),
                author     = doc_meta.get("author", "Unknown"),
                year       = doc_meta.get("year",   "Unknown"),
                doc_type   = doc_meta.get("type",   "publication"),
                text       = sample,
                max_claims = DISTILL_MAX_CLAIMS,
            ),
            timeout_seconds=120,
        )
        if response is None:
            return []
        raw = strip_think_tags(response)
    except Exception as e:
        log.warning(f"  Distillation failed: {e}")
        return []

    now    = datetime.now().isoformat(timespec="seconds")
    claims = []
    entries = re.split(r'\n\s*\d+[\.\)]\s+', '\n' + raw)
    for entry in entries:
        entry = re.sub(r'\s*\n\s*', ' ', entry).strip()
        entry = re.sub(r'^[-*]\s*', '', entry).strip()
        if entry.count('.') < 2 or len(entry) < 80:
            continue
        claims.append({
            "claim":       entry,
            "doc_id":      doc_meta.get("id",     ""),
            "title":       doc_meta.get("title",  ""),
            "author":      doc_meta.get("author", ""),
            "year":        doc_meta.get("year",   ""),
            "type":        doc_meta.get("type",   ""),
            "pdf_url":     doc_meta.get("pdf_url",""),
            "doc_label":   "",
            "doc_summary": "",
            "extracted":   now,
        })

    log.info(f"  Distilled {len(claims)} claims")

    if claims:
        claims_text = "\n".join(f"- {c['claim']}" for c in claims[:10])
        try:
            sum_response = _invoke_with_timeout(
                llm,
                SUMMARY_PROMPT.format(
                    title       = doc_meta.get("title",  "Unknown"),
                    author      = doc_meta.get("author", "Unknown"),
                    year        = doc_meta.get("year",   "Unknown"),
                    claims_text = claims_text,
                ),
                timeout_seconds=60,
            )
            sum_raw = strip_think_tags(sum_response) if sum_response else ""
        except Exception as e:
            log.warning(f"  Summary failed: {e}")
            sum_raw = ""

        doc_label = doc_summary = ""
        for line in sum_raw.splitlines():
            line = line.strip()
            if line.upper().startswith("TITLE:"):
                doc_label = line[6:].strip()
            elif line.upper().startswith("SUMMARY:"):
                doc_summary = line[8:].strip()

        log.info(f"  Label: {doc_label}")
        for c in claims:
            c["doc_label"]   = doc_label
            c["doc_summary"] = doc_summary

    return claims

# ─────────────────────────────────────────────
# STAGE 5 - EXPORT
# ─────────────────────────────────────────────

def export_ideas() -> None:
    progress   = load_json(PROGRESS_FILE, {})
    all_claims = [c for rec in progress.values() for c in rec.get("claims", [])]

    if not all_claims:
        log.warning("No claims to export - run the pipeline first.")
        return

    fieldnames = [
        "claim", "doc_label", "doc_summary",
        "doc_id", "title", "author", "year", "type", "pdf_url", "extracted",
    ]
    with open(IDEAS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_claims)

    log.info(f"Exported {len(all_claims)} claims -> {IDEAS_CSV}")
    print(f"\n{len(all_claims)} claims -> {IDEAS_CSV}")

# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

def process_one(
    doc_meta: dict,
    pdf_path: Path,
    client:   QdrantClient,
    llm:      ChatOllama,
) -> tuple[list[dict], list[dict]]:
    """Run the full pipeline for one document. Returns (claims, image_refs)."""
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {doc_meta.get('title','?')}")
    log.info(f"  Author: {doc_meta.get('author','?')}  Year: {doc_meta.get('year','?')}")

    chunks, image_refs = extract_pdf(pdf_path, doc_meta)
    embed_chunks_with_deduplication(chunks, doc_meta, image_refs, client, llm)
    claims = distil_document(chunks, doc_meta, llm)
    return claims, image_refs


def download_pdf(url: str, dest_dir: Path, filename: str) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        log.info(f"  Already downloaded: {dest.name}")
        return dest
    try:
        log.info(f"  Downloading {url}")
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return dest
    except Exception as exc:
        log.error(f"  Download failed: {exc}")
        return None

# ─────────────────────────────────────────────
# CLI COMMANDS
# ─────────────────────────────────────────────

def cmd_crawl(_args) -> None:
    manifest = crawl_library()
    print(f"\nManifest: {len(manifest)} PDFs -> {MANIFEST_FILE}")


def cmd_run(args) -> None:
    manifest = load_json(MANIFEST_FILE, None)
    if manifest is None:
        print("No manifest found. Run:  python knowledge_pipeline.py crawl")
        return

    progress         = load_json(PROGRESS_FILE, {})
    processed_titles = load_json(PROCESSED_TITLES, [])

    client = get_qdrant()
    ensure_collection(client)
    llm    = get_llm()

    pdfs_dir = DATA_DIR / "pdfs"
    pdfs_dir.mkdir(exist_ok=True)

    limit = getattr(args, "limit", None)
    done  = 0

    for entry in manifest:
        if limit and done >= limit:
            break

        doc_id = entry["id"]
        title  = entry["title"]

        # Already processed in a previous run
        if doc_id in progress:
            log.debug(f"Already done: {title[:60]}")
            continue

        # Title seen before from a different source - skip
        if title in processed_titles:
            log.info(f"Title already in knowledge base, skipping: {title[:60]}")
            progress[doc_id] = {"skipped_title": True}
            save_json(PROGRESS_FILE, progress)
            continue

        pdf_path = download_pdf(
            entry["pdf_url"], pdfs_dir, _safe_name(title) + ".pdf"
        )
        if not pdf_path:
            progress[doc_id] = {"error": "download_failed"}
            save_json(PROGRESS_FILE, progress)
            continue

        time.sleep(CRAWL_DELAY)

        try:
            claims, _ = process_one(entry, pdf_path, client, llm)
            progress[doc_id] = {
                "title":     title,
                "author":    entry["author"],
                "year":      entry["year"],
                "claims":    claims,
                "completed": datetime.now().isoformat(timespec="seconds"),
            }
            if title not in processed_titles:
                processed_titles.append(title)
            save_json(PROGRESS_FILE,    progress)
            save_json(PROCESSED_TITLES, processed_titles)
            done += 1
            log.info(f"  Done ({done} this session)")

        except Exception as exc:
            log.error(f"  Failed: {exc}", exc_info=True)
            progress[doc_id] = {"error": str(exc)}
            save_json(PROGRESS_FILE, progress)

    total = sum(1 for v in progress.values() if "claims" in v)
    print(f"\nSession: {done} processed this run, {total} total in knowledge base.")


def cmd_single(args) -> None:
    pdf_path = Path(args.path)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    doc_meta = {
        "id":      hashlib.md5(str(pdf_path).encode()).hexdigest()[:10],
        "title":   args.title  or pdf_path.stem,
        "author":  args.author or "Unknown",
        "year":    args.year   or "Unknown",
        "type":    args.type   or "publication",
        "pdf_url": str(pdf_path),
    }

    client = get_qdrant()
    ensure_collection(client)
    llm    = get_llm()

    claims, image_refs = process_one(doc_meta, pdf_path, client, llm)

    if claims:
        print(f"\n{'='*60}")
        print(f"LABEL:   {claims[0].get('doc_label','')}")
        print(f"SUMMARY: {claims[0].get('doc_summary','')}")

    print(f"\n{'='*60}")
    print(f"DISTILLED CLAIMS ({len(claims)}):")
    print(f"{'='*60}")
    for i, c in enumerate(claims, 1):
        print(f"{i:2}. {c['claim']}")

    if claims:
        out_csv = DATA_DIR / f"ideas_{doc_meta['id']}.csv"
        fieldnames = [
            "claim", "doc_label", "doc_summary",
            "doc_id", "title", "author", "year", "type", "pdf_url", "extracted",
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(claims)
        print(f"\nClaims  -> {out_csv}")

    if image_refs:
        img_csv = DATA_DIR / f"images_{doc_meta['id']}.csv"
        with open(img_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["doc_id","doc_title","page","img_index","path","caption"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(image_refs)
        print(f"Images  -> {img_csv}")

        captioned = [r for r in image_refs if r.get("caption")]
        print(f"\nContent images saved: {len(image_refs)}")
        if captioned:
            print(f"Captions found ({len(captioned)}/{len(image_refs)}):")
            for r in captioned:
                print(f"  p.{r['page']} img {r['img_index']}: {r['caption'][:100]}")
        else:
            print("No captions detected.")


def cmd_export(_args) -> None:
    export_ideas()


def cmd_purge(_args) -> None:
    import shutil
    confirm = input(
        "This will delete the vector DB, progress, and processed titles.\n"
        "Type YES to confirm: "
    )
    if confirm.strip().upper() != "YES":
        print("Cancelled.")
        return

    if QDRANT_URL:
        try:
            client = get_qdrant()
            client.delete_collection(QDRANT_COLLECTION)
            print(f"Dropped remote collection '{QDRANT_COLLECTION}' on {QDRANT_URL}")
        except Exception as exc:
            print(f"Could not drop remote collection: {exc}")
    elif QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH)
        print(f"Deleted {QDRANT_PATH}")

    for f in [PROGRESS_FILE, PROCESSED_TITLES]:
        if f.exists():
            f.unlink()
            print(f"Deleted {f}")

    print("Purged. Ready for a clean run.")


def cmd_status(_args) -> None:
    manifest  = load_json(MANIFEST_FILE,  [])
    progress  = load_json(PROGRESS_FILE,  {})
    titles    = load_json(PROCESSED_TITLES, [])

    total        = len(manifest)
    completed    = sum(1 for v in progress.values() if "claims"        in v)
    errors       = sum(1 for v in progress.values() if "error"         in v)
    skipped      = sum(1 for v in progress.values() if "skipped_title" in v)
    remaining    = total - len(progress)
    total_claims = sum(len(v.get("claims", [])) for v in progress.values())

    try:
        client   = get_qdrant()
        ensure_collection(client)
        qdrant_n = collection_count(client)
    except Exception:
        qdrant_n = "unavailable"

    print(f"\n{'='*44}")
    print(f"  KNOWLEDGE PIPELINE STATUS")
    print(f"{'='*44}")
    print(f"  Library URL    : {LIBRARY_URL or '(not set)'}")
    print(f"  Manifest PDFs  : {total}")
    print(f"  Completed      : {completed}")
    print(f"  Errors         : {errors}")
    print(f"  Skipped(title) : {skipped}")
    print(f"  Remaining      : {remaining}")
    print(f"  Total claims   : {total_claims}")
    print(f"  Unique titles  : {len(titles)}")
    print(f"  Qdrant vectors : {qdrant_n}")
    print(f"{'='*44}\n")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("crawl", help="Stage 1: crawl library, build manifest")

    p_run = sub.add_parser("run", help="Stages 2-4: process PDFs")
    p_run.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N unprocessed documents this session",
    )

    p_single = sub.add_parser("single", help="Test pipeline on a local PDF")
    p_single.add_argument("path", help="Path to PDF file")
    p_single.add_argument("--title",  default=None)
    p_single.add_argument("--author", default=None)
    p_single.add_argument("--year",   default=None)
    p_single.add_argument(
        "--type", default=None,
        help="publication / book / article / field_report / thesis",
    )

    sub.add_parser("export", help="Stage 5: export all claims to CSV")
    sub.add_parser("purge",  help="Wipe vector DB and progress files")
    sub.add_parser("status", help="Show progress summary")

    args = parser.parse_args()
    dispatch = {
        "crawl":  cmd_crawl,
        "run":    cmd_run,
        "single": cmd_single,
        "export": cmd_export,
        "purge":  cmd_purge,
        "status": cmd_status,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()