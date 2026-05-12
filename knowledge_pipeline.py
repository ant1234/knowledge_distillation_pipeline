"""
Knowledge Distillation Pipeline
=================================
Crawls any web library, extracts PDFs, chunks and embeds text,
deduplicates via Qdrant vector search, distils novel claims using a
local LLM, and exports a structured knowledge base to CSV.

Set LIBRARY_URL in .env to point at any library index page.

Usage:
  python knowledge_pipeline.py crawl                       # Stage 1: build link manifest
  python knowledge_pipeline.py run                         # Stages 2-4: process all PDFs
  python knowledge_pipeline.py run --limit 3               # Process at most N unprocessed
  python knowledge_pipeline.py single /path/to.pdf         # Test with a local PDF
  python knowledge_pipeline.py folder /path/to/dir         # Process all PDFs in a folder
  python knowledge_pipeline.py folder /path/to/dir --tier 2  # Tag with tier number
  python knowledge_pipeline.py export                      # Stage 5: export claims to CSV
  python knowledge_pipeline.py export_functions            # Export functional/speculative claims
  python knowledge_pipeline.py purge                       # Wipe DB and progress
  python knowledge_pipeline.py status                      # Show progress summary
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

DISTILL_MAX_CLAIMS   = int(os.getenv("DISTILL_MAX_CLAIMS",  "40"))
# Words sent to the LLM per distillation batch.
# Every word of every document is processed — no truncation.
# 6000 words ≈ one batch call of ~2 minutes for deepseek-r1:14b.
DISTILL_BATCH_WORDS  = int(os.getenv("DISTILL_BATCH_WORDS", "6000"))

TEMPERATURE          = float(os.getenv("TEMPERATURE",       "0.3"))
MAX_TOKENS           = int(os.getenv("MAX_TOKENS",          "8192"))
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

# ── Split progress architecture ──────────────
# PROGRESS_FILE : tiny index loaded at every session start
#                 stores only doc_id -> status, no claims
# CLAIMS_FILE   : append-only claims store, only read at export time
PROGRESS_FILE        = DATA_DIR / "progress_index.json"
CLAIMS_FILE          = DATA_DIR / "claims_store.json"
PROCESSED_TITLES     = DATA_DIR / "processed_titles.json"

IDEAS_CSV            = DATA_DIR / "ideas.csv"
FUNCTIONS_CSV        = DATA_DIR / "functional_claims.csv"
LOG_FILE             = DATA_DIR / "pipeline.log"

EMBEDDING_DIM        = 768
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
    words = text.split()
    if len(words) > CHUNK_SIZE:
        text = " ".join(words[:CHUNK_SIZE])
    if len(text) > MAX_CHUNK_CHARS:
        text = text[:MAX_CHUNK_CHARS]
    return get_embeddings().embed_query(text)

def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _invoke_with_timeout(llm, prompt, timeout_seconds=120):
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
        log.warning(f"  LLM call timed out after {timeout_seconds}s - skipping")
        return None
    if error[0]:
        raise error[0]
    return result[0]

# ─────────────────────────────────────────────
# QDRANT
# ─────────────────────────────────────────────

def get_qdrant() -> QdrantClient:
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

def append_claims_to_store(doc_id: str, claims: list, functional_claims: list) -> None:
    """
    Append or update a single document's claims in claims_store.json.
    Only loads the whole file once per call - efficient for large stores.
    """
    store = load_json(CLAIMS_FILE, {})
    store[doc_id] = {
        "claims":            claims,
        "functional_claims": functional_claims,
    }
    save_json(CLAIMS_FILE, store)

# ─────────────────────────────────────────────
# STAGE 1 - CRAWL
# ─────────────────────────────────────────────

def crawl_library() -> list[dict]:
    if not LIBRARY_URL:
        raise ValueError("LIBRARY_URL is not set. Add it to your .env file.")

    log.info(f"Crawling: {LIBRARY_URL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "KnowledgePipelineBot/1.0 (academic research)"})

    resp = session.get(LIBRARY_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    from urllib.parse import urlparse
    parsed   = urlparse(LIBRARY_URL)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    doc_links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = href if href.startswith("http") else base_url + href
        if full.lower().endswith(".pdf"):
            if full not in doc_links:
                doc_links.append(full)
            continue
        if "/library/" in full and full != LIBRARY_URL and full not in doc_links:
            doc_links.append(full)

    log.info(f"Found {len(doc_links)} candidate document pages")

    existing  = load_json(MANIFEST_FILE, [])
    seen_pdfs = {e["pdf_url"] for e in existing}
    manifest  = list(existing)

    for idx, doc_url in enumerate(doc_links):
        try:
            log.info(f"[{idx+1}/{len(doc_links)}] {doc_url}")

            if doc_url.lower().endswith(".pdf"):
                pdf_url = doc_url
                if pdf_url in seen_pdfs:
                    continue
                seen_pdfs.add(pdf_url)
                title  = Path(pdf_url).stem.replace("_", " ").replace("-", " ").title()
                entry = {
                    "id":        hashlib.md5(pdf_url.encode()).hexdigest()[:10],
                    "title":     title.strip(),
                    "author":    "Unknown",
                    "year":      _extract_year_from_url(pdf_url),
                    "type":      "publication",
                    "tier":      1,
                    "pdf_url":   pdf_url,
                    "page_url":  pdf_url,
                    "processed": False,
                }
                manifest.append(entry)
                save_json(MANIFEST_FILE, manifest)
                log.info(f"  + {title[:70]} [publication]")
                time.sleep(0.5)
                continue

            time.sleep(CRAWL_DELAY)
            r = session.get(doc_url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                log.warning(f"  Skipped (HTTP {r.status_code})")
                continue

            dsoup   = BeautifulSoup(r.text, "html.parser")
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
                "id":        hashlib.md5(pdf_url.encode()).hexdigest()[:10],
                "title":     title.strip(),
                "author":    author.strip(),
                "year":      year,
                "type":      dtype,
                "tier":      1,
                "pdf_url":   pdf_url,
                "page_url":  doc_url,
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
    m = re.search(r'(1[89]\d{2}|20[012]\d)', url)
    return m.group(0) if m else "Unknown"

def _extract_year_from_filename(filename: str) -> str:
    m = re.search(r'(1[89]\d{2}|20[012]\d)', filename)
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
    if width == 0 or height == 0:
        return False
    ratio = width / height
    for pr in _PAGE_RATIOS:
        if abs(ratio - pr) < PAGE_RATIO_TOLERANCE and width * height > 500_000:
            return True
    return False

def _extract_caption_near_image(page, img_xref: int) -> str:
    try:
        img_rect: Optional[fitz.Rect] = None
        for item in page.get_image_info(xrefs=True):
            if item.get("xref") == img_xref:
                bbox = item.get("bbox")
                if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    img_rect = fitz.Rect(bbox)
                break
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
        if _LAYOUT_AVAILABLE:
            page_text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE,
                sort=True,
            )
        else:
            page_text = page.get_text("text")

        try:
            for t_idx, table in enumerate(page.find_tables()):
                df = __import__('pandas').DataFrame(table.extract())
                page_text += (
                    f"\n[TABLE p.{page_num+1} t.{t_idx+1}]\n"
                    + df.to_string(index=False, header=False)
                    + "\n"
                )
        except Exception:
            pass

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
                if img_w * img_h < MIN_IMAGE_PIXELS:
                    tiny_discarded += 1
                    continue
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
    log.info(f"  Extracted {len(chunks)} chunks, {len(image_records)} content images from {pdf_path.name}")
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
# STAGE 3 - DEDUPLICATE + EMBED
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

    safe_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) > CHUNK_SIZE:
            chunk = " ".join(words[:CHUNK_SIZE])
        if len(chunk) > MAX_CHUNK_CHARS:
            chunk = chunk[:MAX_CHUNK_CHARS]
        safe_chunks.append(chunk)

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
    support_updates: dict[int, dict] = {}

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
            existing_point_id = hits[0].id
            existing_doc_id   = hits[0].payload.get("doc_id", "")
            if existing_doc_id != doc_meta.get("id", ""):
                if existing_point_id not in support_updates:
                    support_updates[existing_point_id] = {
                        "existing_payload": hits[0].payload,
                        "supporting_sources": [],
                    }
                support_updates[existing_point_id]["supporting_sources"].append({
                    "doc_id": doc_meta.get("id",    ""),
                    "title":  doc_meta.get("title", ""),
                    "year":   doc_meta.get("year",  ""),
                    "tier":   doc_meta.get("tier",  1),
                })
            skipped += 1

        elif similarity >= SIMILARITY_CHECK:
            if llm_checks >= MAX_LLM_CHECKS_PER_DOC:
                points.append(_make_point(chunk, vector, doc_meta))
                added += 1
            else:
                llm_checks += 1
                best_chunk = hits[0].payload.get("text", "")
                if _llm_duplicate_check(chunk, best_chunk, llm):
                    existing_point_id = hits[0].id
                    existing_doc_id   = hits[0].payload.get("doc_id", "")
                    if existing_doc_id != doc_meta.get("id", ""):
                        if existing_point_id not in support_updates:
                            support_updates[existing_point_id] = {
                                "existing_payload": hits[0].payload,
                                "supporting_sources": [],
                            }
                        support_updates[existing_point_id]["supporting_sources"].append({
                            "doc_id": doc_meta.get("id",    ""),
                            "title":  doc_meta.get("title", ""),
                            "year":   doc_meta.get("year",  ""),
                            "tier":   doc_meta.get("tier",  1),
                        })
                    skipped += 1
                else:
                    points.append(_make_point(chunk, vector, doc_meta))
                    added += 1
        else:
            points.append(_make_point(chunk, vector, doc_meta))
            added += 1

    for point_id, update_data in support_updates.items():
        try:
            existing_payload = update_data["existing_payload"]
            new_sources      = update_data["supporting_sources"]
            current_count    = existing_payload.get("support_count", 0)
            current_sources  = existing_payload.get("supporting_sources", [])
            existing_doc_ids = {s["doc_id"] for s in current_sources}
            unique_new = [s for s in new_sources if s["doc_id"] not in existing_doc_ids]
            if unique_new:
                updated_payload = dict(existing_payload)
                updated_payload["support_count"]      = current_count + len(unique_new)
                updated_payload["supporting_sources"] = current_sources + unique_new
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload=updated_payload,
                    points=[point_id],
                )
        except Exception as e:
            log.debug(f"  Support update failed for point {point_id}: {e}")

    if points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    support_count = sum(len(v["supporting_sources"]) for v in support_updates.values())
    log.info(
        f"  Chunks: {added} added, {skipped} skipped "
        f"({support_count} support links recorded), {llm_checks} LLM checks"
    )
    return added, skipped, llm_checks

def _make_point(chunk: str, vector: list[float], meta: dict) -> PointStruct:
    uid = hashlib.md5((chunk + meta.get("id", "")).encode()).hexdigest()
    return PointStruct(
        id=int(uid[:8], 16),
        vector=vector,
        payload={
            "text":               chunk,
            "doc_id":             meta.get("id",     ""),
            "title":              meta.get("title",  ""),
            "author":             meta.get("author", ""),
            "year":               meta.get("year",   ""),
            "type":               meta.get("type",   ""),
            "tier":               meta.get("tier",   1),
            "support_count":      0,
            "supporting_sources": [],
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
        return False
    return strip_think_tags(response).strip().upper().startswith("Y")

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

Task: Identify up to {max_claims} DISTINCT factual claims, measurements, or \
interpretations that THIS document specifically contributes. You MUST name the \
specific subject by name in EVERY sentence of EVERY claim - never write "the tomb", \
"the structure", "the researchers", "this finding", or any other vague reference \
without the specific name.

CRITICAL GROUNDING RULES - these override everything else:
- ONLY extract claims that are EXPLICITLY stated in the TEXT above.
- Do NOT use your training knowledge. Do NOT infer. Do NOT guess.
- If the TEXT does not clearly state a fact, omit it entirely.
- Every measurement, date, name, and location MUST appear verbatim in the TEXT.
- If the TEXT contains no extractable factual claims, return the single word: NONE

ANTI-DUPLICATION RULES - strictly enforced:
- Every claim must be about a DIFFERENT specific finding or measurement.
- Do NOT repeat the same fact in multiple claims.
- Do NOT rephrase a claim you have already written.
- Stop early if you run out of distinct claims rather than padding with repetitions.

Formatting rules:
- Each claim MUST be exactly 2-4 complete sentences of continuous flowing prose.
- NO bullet points, dashes, asterisks, or sub-items within a claim.
- Every sentence must name the specific subject.
- Sentence 1: name the subject AND state the finding with numbers/measurements/dates from TEXT.
- Sentence 2: further context about that same named subject from TEXT.
- Sentence 3 (if relevant): method used or significance, as stated in TEXT.
- If text contains [FIGURE CAPTION] markers, incorporate what the figure shows.
- EXCLUDE claims that only cite prior work without adding new data.
- EXCLUDE vague statements with no specific data, numbers, or named subjects.
- EXCLUDE any claim shorter than 2 full sentences.
- Start each claim with its number followed by a period.
- Each numbered claim must be on its own line.

Return ONLY a numbered list where each item is 2-4 sentences of flowing prose, \
OR the single word NONE if no extractable claims exist. \
No bullets. No dashes. No preamble. No commentary after the list.
"""

SUMMARY_PROMPT = """\
You are a research analyst extracting structured knowledge from academic publications.

Source: {title} - {author} ({year})

Top claims from this document:
{claims_text}

CRITICAL: Your title and summary must be based ONLY on the claims listed above.

Tasks:
1. Write a SHORT descriptive title (8 words maximum) that names the specific subject.
2. Write a 2-3 sentence summary naming the specific subject in the first sentence.

Respond in this exact format and nothing else:
TITLE: <title>
SUMMARY: <summary>
"""

FUNCTION_PROMPT = """\
You are a research analyst extracting claims about the PURPOSE, FUNCTION, or USE \
of ancient Egyptian structures, particularly the Giza pyramids and related monuments.

Source document:
  Title:  {title}
  Author: {author}
  Year:   {year}
  Type:   {doc_type}

TEXT:
{text}

Task: Identify every claim, theory, speculation, or assertion this document makes \
about WHAT the pyramids or Giza structures were FOR, HOW they worked, or WHAT \
PURPOSE they served. Include mainstream, alternative, and speculative claims. \
Include claims the author endorses AND claims the author merely mentions or debates.

CRITICAL GROUNDING RULES:
- ONLY extract claims present in the TEXT above.
- Do NOT use training knowledge. Do NOT infer. Do NOT fabricate.
- If the TEXT contains NO claims about function or purpose, return: NONE
- "The pyramid served as a tomb" is valid if stated in the TEXT.

What counts: what it was built for, how a feature functioned, what astronomical/
acoustic/hydraulic/spiritual/engineering purpose it served.

What does NOT count: pure measurements without stated purpose, historical facts
about who built it unless linked to a purpose claim, conservation methodology.

Output rules:
- Each claim must be ONE clear sentence stating what the structure was for/how it worked.
- Name the specific structure in every claim.
- Attribute: "According to {author}," if endorsed, or "The text notes the theory that" if reported.
- Include speculative and fringe claims if present.
- Start each claim with its number followed by a period.

Return ONLY a numbered list of single-sentence functional claims, \
OR the single word NONE if no functional claims exist. \
No preamble. No commentary.
"""


def distil_document(
    chunks: list[str],
    doc_meta: dict,
    llm: ChatOllama,
) -> tuple[list[dict], list[dict]]:
    """
    Distil ALL chunks into factual and functional claims.

    The document is split into batches of DISTILL_BATCH_WORDS words and every
    batch is processed separately. This means no content is ever left behind
    regardless of document length — a 3000-page book is fully processed.

    Cross-batch deduplication is handled by fingerprint sets so the same
    claim is never recorded twice even if it appears in multiple batches.

    Returns (claims, functional_claims).
    """
    if not chunks:
        log.info("  No chunks to distil")
        return [], []

    now = datetime.now().isoformat(timespec="seconds")

    # ── Split all document words into fixed-size batches ───────────────────
    # No truncation — the while loop runs until every word is processed.
    all_words   = " ".join(chunks).split()
    total_words = len(all_words)
    batches: list[str] = []
    i = 0
    while i < total_words:
        batches.append(" ".join(all_words[i : i + DISTILL_BATCH_WORDS]))
        i += DISTILL_BATCH_WORDS

    log.info(
        f"  Distilling {total_words:,} words across "
        f"{len(batches)} batch(es) of ~{DISTILL_BATCH_WORDS:,} words"
    )

    # ── Per-document deduplication sets ────────────────────────────────────
    # Fingerprints are checked across ALL batches so a repeated claim in
    # batch 3 that already appeared in batch 1 is silently dropped.
    all_claim_texts: list[str] = []
    all_func_texts:  list[str] = []
    seen_claim_fps:  set[str]  = set()
    seen_func_fps:   set[str]  = set()

    for batch_num, batch_text in enumerate(batches, 1):
        log.info(f"  Batch {batch_num}/{len(batches)}...")

        # ── Factual claims for this batch ──────────────────────────────────
        try:
            response = _invoke_with_timeout(
                llm,
                DISTIL_PROMPT.format(
                    title      = doc_meta.get("title",  "Unknown"),
                    author     = doc_meta.get("author", "Unknown"),
                    year       = doc_meta.get("year",   "Unknown"),
                    doc_type   = doc_meta.get("type",   "publication"),
                    text       = batch_text,
                    max_claims = DISTILL_MAX_CLAIMS,
                ),
                timeout_seconds=120,
            )
            if response is not None:
                raw = strip_think_tags(response).strip()
                if raw.upper() != "NONE":
                    for entry in re.split(r'\n\s*\d+[\.\)]\s+', '\n' + raw):
                        entry = re.sub(r'\s*\n\s*', ' ', entry).strip()
                        entry = re.sub(r'^[-*]\s*', '', entry).strip()
                        if entry.count('.') < 2 or len(entry) < 80:
                            continue
                        fp = entry[:120].lower()
                        if fp not in seen_claim_fps:
                            seen_claim_fps.add(fp)
                            all_claim_texts.append(entry)
        except Exception as e:
            log.warning(f"  Batch {batch_num} factual distillation failed: {e}")

        # ── Functional claims for this batch ───────────────────────────────
        try:
            func_response = _invoke_with_timeout(
                llm,
                FUNCTION_PROMPT.format(
                    title    = doc_meta.get("title",  "Unknown"),
                    author   = doc_meta.get("author", "Unknown"),
                    year     = doc_meta.get("year",   "Unknown"),
                    doc_type = doc_meta.get("type",   "publication"),
                    text     = batch_text,
                ),
                timeout_seconds=120,
            )
            if func_response is not None:
                func_raw = strip_think_tags(func_response).strip()
                if func_raw.upper() != "NONE":
                    for line in func_raw.splitlines():
                        line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                        if len(line) < 30:
                            continue
                        fp = line[:100].lower()
                        if fp not in seen_func_fps:
                            seen_func_fps.add(fp)
                            all_func_texts.append(line)
        except Exception as e:
            log.warning(f"  Batch {batch_num} functional distillation failed: {e}")

    log.info(
        f"  Distilled {len(all_claim_texts)} claims, "
        f"{len(all_func_texts)} functional claims "
        f"across {len(batches)} batch(es)"
    )

    # ── Build claim dicts ──────────────────────────────────────────────────
    claims: list[dict] = [
        {
            "claim":              text,
            "doc_id":             doc_meta.get("id",     ""),
            "title":              doc_meta.get("title",  ""),
            "author":             doc_meta.get("author", ""),
            "year":               doc_meta.get("year",   ""),
            "type":               doc_meta.get("type",   ""),
            "tier":               doc_meta.get("tier",   1),
            "pdf_url":            doc_meta.get("pdf_url",""),
            "doc_label":          "",
            "doc_summary":        "",
            "extracted":          now,
            "support_count":      0,
            "supporting_sources": "",
        }
        for text in all_claim_texts
    ]

    # ── Generate document summary from first 10 claims ─────────────────────
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

    # ── Build functional claim dicts ───────────────────────────────────────
    functional_claims: list[dict] = [
        {
            "claim":     text,
            "doc_id":    doc_meta.get("id",     ""),
            "title":     doc_meta.get("title",  ""),
            "author":    doc_meta.get("author", ""),
            "year":      doc_meta.get("year",   ""),
            "type":      doc_meta.get("type",   ""),
            "tier":      doc_meta.get("tier",   1),
            "pdf_url":   doc_meta.get("pdf_url",""),
            "extracted": now,
        }
        for text in all_func_texts
    ]

    log.info(f"  Functional claims: {len(functional_claims)}")
    return claims, functional_claims

# ─────────────────────────────────────────────
# STAGE 5 - EXPORT
# ─────────────────────────────────────────────

def export_ideas() -> None:
    store      = load_json(CLAIMS_FILE, {})
    all_claims = [c for rec in store.values() for c in rec.get("claims", [])]

    if not all_claims:
        log.warning("No claims to export - run the pipeline first.")
        return

    log.info("  Enriching claims with support data from Qdrant...")
    try:
        client   = get_qdrant()
        enriched = 0
        emb      = get_embeddings()
        for claim in all_claims:
            try:
                claim_text = claim.get("claim", "")
                if not claim_text:
                    continue
                words = claim_text.split()
                if len(words) > CHUNK_SIZE:
                    claim_text = " ".join(words[:CHUNK_SIZE])
                if len(claim_text) > MAX_CHUNK_CHARS:
                    claim_text = claim_text[:MAX_CHUNK_CHARS]
                vector  = emb.embed_query(claim_text)
                results = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=vector,
                    limit=1,
                )
                if results.points:
                    hit = results.points[0]
                    if hit.payload.get("doc_id") == claim.get("doc_id"):
                        sc = hit.payload.get("support_count", 0)
                        ss = hit.payload.get("supporting_sources", [])
                        claim["support_count"] = sc
                        claim["supporting_sources"] = "|".join(
                            f"{s.get('title','?')} ({s.get('year','?')})"
                            for s in ss
                        )
                        if sc > 0:
                            enriched += 1
            except Exception:
                pass
        log.info(f"  {enriched} claims enriched with support data")
    except Exception as e:
        log.warning(f"  Could not enrich with support data: {e}")

    fieldnames = [
        "claim", "doc_label", "doc_summary",
        "doc_id", "title", "author", "year", "type", "tier",
        "support_count", "supporting_sources",
        "pdf_url", "extracted",
    ]
    with open(IDEAS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_claims)

    total_support = sum(c.get("support_count", 0) for c in all_claims)
    top_claims    = sorted(all_claims, key=lambda c: c.get("support_count", 0), reverse=True)[:5]

    log.info(f"Exported {len(all_claims)} claims -> {IDEAS_CSV}")
    print(f"\n{len(all_claims)} claims -> {IDEAS_CSV}")
    print(f"Total cross-document support links: {total_support}")
    if top_claims and top_claims[0].get("support_count", 0) > 0:
        print(f"\nTop 5 most-supported claims:")
        for c in top_claims:
            sc = c.get("support_count", 0)
            if sc > 0:
                print(f"  [{sc} sources] {c['claim'][:120]}...")

def export_functional_claims() -> None:
    store      = load_json(CLAIMS_FILE, {})
    all_claims = [c for rec in store.values() for c in rec.get("functional_claims", [])]

    if not all_claims:
        log.warning("No functional claims to export.")
        print("No functional claims found.")
        return

    fieldnames = [
        "claim", "doc_id", "title", "author", "year",
        "type", "tier", "pdf_url", "extracted",
    ]
    with open(FUNCTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_claims)

    log.info(f"Exported {len(all_claims)} functional claims -> {FUNCTIONS_CSV}")
    print(f"\n{len(all_claims)} functional claims -> {FUNCTIONS_CSV}")

# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

def process_one(
    doc_meta: dict,
    pdf_path: Path,
    client:   QdrantClient,
    llm:      ChatOllama,
) -> tuple[list[dict], list[dict], list[dict]]:
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {doc_meta.get('title','?')}")
    log.info(f"  Author: {doc_meta.get('author','?')}  Year: {doc_meta.get('year','?')}  Tier: {doc_meta.get('tier',1)}")
    chunks, image_refs        = extract_pdf(pdf_path, doc_meta)
    embed_chunks_with_deduplication(chunks, doc_meta, image_refs, client, llm)
    claims, functional_claims = distil_document(chunks, doc_meta, llm)
    return claims, functional_claims, image_refs

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

        if doc_id in progress:
            if progress[doc_id].get("error") == "download_failed":
                log.info(f"Retrying failed download: {title[:60]}")
            else:
                log.debug(f"Already done: {title[:60]}")
                continue

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
            claims, functional_claims, _ = process_one(entry, pdf_path, client, llm)

            progress[doc_id] = {
                "title":      title,
                "year":       entry["year"],
                "tier":       entry.get("tier", 1),
                "completed":  datetime.now().isoformat(timespec="seconds"),
                "has_claims": len(claims) > 0,
            }
            save_json(PROGRESS_FILE, progress)
            append_claims_to_store(doc_id, claims, functional_claims)

            if title not in processed_titles:
                processed_titles.append(title)
            save_json(PROCESSED_TITLES, processed_titles)

            done += 1
            log.info(f"  Done ({done} this session)")

        except Exception as exc:
            log.error(f"  Failed: {exc}", exc_info=True)
            progress[doc_id] = {"error": str(exc)}
            save_json(PROGRESS_FILE, progress)

    total = sum(1 for v in progress.values() if v.get("completed"))
    print(f"\nSession: {done} processed this run, {total} total in knowledge base.")


def cmd_folder(args) -> None:
    folder = Path(args.folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return

    tier  = getattr(args, "tier",  1)
    limit = getattr(args, "limit", None)

    pdf_files = sorted(folder.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder}")
        return

    log.info(f"Found {len(pdf_files)} PDF(s) in {folder} (Tier {tier})")

    progress         = load_json(PROGRESS_FILE, {})
    processed_titles = load_json(PROCESSED_TITLES, [])

    client = get_qdrant()
    ensure_collection(client)
    llm    = get_llm()

    done = 0

    for pdf_path in pdf_files:
        if limit and done >= limit:
            break

        stem   = pdf_path.stem
        title  = stem.replace("_", " ").replace("-", " ").title()
        year   = _extract_year_from_filename(stem)
        doc_id = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:10]

        if doc_id in progress:
            if progress[doc_id].get("error") == "download_failed":
                log.info(f"Retrying: {title[:60]}")
            else:
                log.debug(f"Already done: {title[:60]}")
                continue

        if title in processed_titles:
            log.info(f"Title already in knowledge base, skipping: {title[:60]}")
            progress[doc_id] = {"skipped_title": True}
            save_json(PROGRESS_FILE, progress)
            continue

        doc_meta = {
            "id":      doc_id,
            "title":   title,
            "author":  "Unknown",
            "year":    year,
            "type":    "publication",
            "tier":    tier,
            "pdf_url": str(pdf_path.resolve()),
        }

        log.info(f"  [{done+1}] {pdf_path.name}")

        try:
            claims, functional_claims, _ = process_one(doc_meta, pdf_path, client, llm)

            progress[doc_id] = {
                "title":      title,
                "year":       year,
                "tier":       tier,
                "completed":  datetime.now().isoformat(timespec="seconds"),
                "has_claims": len(claims) > 0,
            }
            save_json(PROGRESS_FILE, progress)
            append_claims_to_store(doc_id, claims, functional_claims)

            if title not in processed_titles:
                processed_titles.append(title)
            save_json(PROCESSED_TITLES, processed_titles)

            done += 1
            log.info(f"  Done ({done} this session)")

        except Exception as exc:
            log.error(f"  Failed on {pdf_path.name}: {exc}", exc_info=True)
            progress[doc_id] = {"error": str(exc)}
            save_json(PROGRESS_FILE, progress)

    total = sum(1 for v in progress.values() if v.get("completed"))
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
        "tier":    getattr(args, "tier", 1),
        "pdf_url": str(pdf_path),
    }

    client = get_qdrant()
    ensure_collection(client)
    llm    = get_llm()

    claims, functional_claims, image_refs = process_one(doc_meta, pdf_path, client, llm)

    if claims:
        print(f"\n{'='*60}")
        print(f"LABEL:   {claims[0].get('doc_label','')}")
        print(f"SUMMARY: {claims[0].get('doc_summary','')}")

    print(f"\n{'='*60}")
    print(f"DISTILLED CLAIMS ({len(claims)}):")
    print(f"{'='*60}")
    for i, c in enumerate(claims, 1):
        print(f"{i:2}. {c['claim']}")

    if functional_claims:
        print(f"\n{'='*60}")
        print(f"FUNCTIONAL CLAIMS ({len(functional_claims)}):")
        print(f"{'='*60}")
        for i, c in enumerate(functional_claims, 1):
            print(f"{i:2}. {c['claim']}")

    if claims:
        out_csv = DATA_DIR / f"ideas_{doc_meta['id']}.csv"
        fieldnames = [
            "claim", "doc_label", "doc_summary",
            "doc_id", "title", "author", "year", "type", "tier",
            "support_count", "supporting_sources", "pdf_url", "extracted",
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
                f, fieldnames=["doc_id","doc_title","page","img_index","path","caption"],
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

def cmd_export_functions(_args) -> None:
    export_functional_claims()


def cmd_purge(_args) -> None:
    import shutil
    confirm = input(
        "This will delete the vector DB, progress index, claims store, and processed titles.\n"
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

    for f in [PROGRESS_FILE, CLAIMS_FILE, PROCESSED_TITLES]:
        if f.exists():
            f.unlink()
            print(f"Deleted {f}")

    print("Purged. Ready for a clean run.")


def cmd_status(_args) -> None:
    manifest  = load_json(MANIFEST_FILE,  [])
    progress  = load_json(PROGRESS_FILE,  {})
    titles    = load_json(PROCESSED_TITLES, [])

    total_claims = 0
    total_func   = 0
    if CLAIMS_FILE.exists():
        store        = load_json(CLAIMS_FILE, {})
        total_claims = sum(len(v.get("claims", []))            for v in store.values())
        total_func   = sum(len(v.get("functional_claims", [])) for v in store.values())

    total       = len(manifest)
    completed   = sum(1 for v in progress.values() if v.get("completed"))
    errors      = sum(1 for v in progress.values() if v.get("error"))
    skipped     = sum(1 for v in progress.values() if v.get("skipped_title"))
    with_claims = sum(1 for v in progress.values() if v.get("has_claims"))
    remaining   = total - len(progress)

    tier_counts: dict[int, int] = {}
    for v in progress.values():
        if v.get("has_claims"):
            t = v.get("tier", 1)
            tier_counts[t] = tier_counts.get(t, 0) + 1

    try:
        client   = get_qdrant()
        ensure_collection(client)
        qdrant_n = collection_count(client)
    except Exception:
        qdrant_n = "unavailable"

    idx_size    = f"{PROGRESS_FILE.stat().st_size/1024:.0f} KB"    if PROGRESS_FILE.exists() else "—"
    claims_size = f"{CLAIMS_FILE.stat().st_size/1024/1024:.1f} MB" if CLAIMS_FILE.exists()   else "—"

    print(f"\n{'='*48}")
    print(f"  KNOWLEDGE PIPELINE STATUS")
    print(f"{'='*48}")
    print(f"  Library URL       : {LIBRARY_URL or '(not set)'}")
    print(f"  Manifest PDFs     : {total}")
    print(f"  Completed         : {completed}")
    print(f"  With claims       : {with_claims}")
    print(f"  Errors            : {errors}")
    print(f"  Skipped(title)    : {skipped}")
    print(f"  Remaining         : {remaining}")
    print(f"  Factual claims    : {total_claims}")
    print(f"  Functional claims : {total_func}")
    print(f"  Unique titles     : {len(titles)}")
    print(f"  Qdrant vectors    : {qdrant_n}")
    print(f"  Batch size        : {DISTILL_BATCH_WORDS:,} words")
    print(f"  Index size        : {idx_size}")
    print(f"  Claims store size : {claims_size}")
    if tier_counts:
        print(f"  By tier           :", end="")
        for t in sorted(tier_counts):
            print(f"  Tier {t}: {tier_counts[t]}", end="")
        print()
    print(f"{'='*48}\n")

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

    p_run = sub.add_parser("run", help="Stages 2-4: process PDFs from manifest")
    p_run.add_argument("--limit", type=int, default=None)

    p_folder = sub.add_parser("folder", help="Process all PDFs in a local folder")
    p_folder.add_argument("folder_path")
    p_folder.add_argument("--tier",  type=int, default=1)
    p_folder.add_argument("--limit", type=int, default=None)

    p_single = sub.add_parser("single", help="Test pipeline on a single local PDF")
    p_single.add_argument("path")
    p_single.add_argument("--title",  default=None)
    p_single.add_argument("--author", default=None)
    p_single.add_argument("--year",   default=None)
    p_single.add_argument("--type",   default=None)
    p_single.add_argument("--tier",   type=int, default=1)

    sub.add_parser("export",           help="Export factual claims to CSV")
    sub.add_parser("export_functions", help="Export functional/speculative claims to CSV")
    sub.add_parser("purge",            help="Wipe DB and progress files")
    sub.add_parser("status",           help="Show progress summary")

    args = parser.parse_args()
    dispatch = {
        "crawl":            cmd_crawl,
        "run":              cmd_run,
        "folder":           cmd_folder,
        "single":           cmd_single,
        "export":           cmd_export,
        "export_functions": cmd_export_functions,
        "purge":            cmd_purge,
        "status":           cmd_status,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()