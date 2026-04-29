"""
Giza Digital Library RAG Pipeline
===================================
Crawls Harvard's Digital Giza Library, extracts PDFs, chunks/embeds text,
deduplicates, distills novel claims, and exports to CSV.

Usage:
  python giza_pipeline.py --help
  python giza_pipeline.py crawl                  # Stage 1: build link manifest
  python giza_pipeline.py run                    # Stages 2-4: process all PDFs
  python giza_pipeline.py run --limit 3          # Process only first 3 unprocessed
  python giza_pipeline.py single /path/file.pdf  # Test with a local PDF
  python giza_pipeline.py export                 # Stage 5: export ideas to CSV
  python giza_pipeline.py purge                  # Wipe vector DB and progress
  python giza_pipeline.py status                 # Show progress summary
"""

import os
import re
import csv
import json
import time
import argparse
import hashlib
import logging
import io as _io
from pathlib import Path
from datetime import datetime
from typing import Optional


import requests
from bs4 import BeautifulSoup
import pymupdf as fitz
from PIL import Image
import pandas as pd

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# CONFIGURATION  (override via environment vars)
# ─────────────────────────────────────────────

LIBRARY_URL         = os.getenv("GIZA_LIBRARY_URL",   "http://giza.fas.harvard.edu/library/")
CRAWL_DELAY         = float(os.getenv("CRAWL_DELAY",  "3"))       # seconds between requests
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT","30"))

CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE",     "400"))   # words per chunk
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP",  "60"))    # word overlap
# nomic-embed-text hard limit is 8192 tokens (~6000 chars conservatively)
MAX_CHUNK_CHARS     = int(os.getenv("MAX_CHUNK_CHARS","5500"))

SIMILARITY_DISCARD  = float(os.getenv("SIMILARITY_DISCARD", "0.85"))  # above = duplicate
SIMILARITY_CHECK    = float(os.getenv("SIMILARITY_CHECK",   "0.70"))  # between = LLM check

DISTILL_MAX_CLAIMS  = int(os.getenv("DISTILL_MAX_CLAIMS",   "20"))    # max claims per doc
TEMPERATURE         = float(os.getenv("TEMPERATURE",        "0.3"))
MAX_TOKENS          = int(os.getenv("MAX_TOKENS",           "4096"))
MODEL_NAME          = os.getenv("MODEL_NAME",  "deepseek-r1:14b")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Image filtering — page-background images are rejected if they match standard
# page aspect ratios (A4 ~0.707, US Letter ~0.773) within this tolerance.
# Also reject images smaller than this pixel count (decorative rules, logos etc.)
PAGE_RATIO_TOLERANCE = float(os.getenv("PAGE_RATIO_TOLERANCE", "0.03"))
MIN_IMAGE_PIXELS     = int(os.getenv("MIN_IMAGE_PIXELS", "10000"))   # ~100×100 px minimum

# Paths
DATA_DIR            = Path(os.getenv("DATA_DIR",    "./giza_data"))
IMAGES_DIR          = DATA_DIR / "images"
VECTORSTORE_DIR     = DATA_DIR / "vectorstore"
MANIFEST_FILE       = DATA_DIR / "manifest.json"
PROGRESS_FILE       = DATA_DIR / "progress.json"
PROCESSED_TITLES    = DATA_DIR / "processed_titles.json"
IDEAS_CSV           = DATA_DIR / "ideas.csv"
LOG_FILE            = DATA_DIR / "pipeline.log"

# Standard page aspect ratios to detect and discard full-page background images
_PAGE_RATIOS = [
    0.7071,   # A4 portrait  (210/297)
    1.4142,   # A4 landscape
    0.7727,   # US Letter portrait  (8.5/11)
    1.2941,   # US Letter landscape
    0.8165,   # A3 portrait
    1.2247,   # A3 landscape
]

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
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LLM + EMBEDDINGS
# ─────────────────────────────────────────────

def get_llm():
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
    )

def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

def strip_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# ─────────────────────────────────────────────
# PERSISTENCE HELPERS
# ─────────────────────────────────────────────

def load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_vectorstore() -> Optional[FAISS]:
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        log.info("Loading existing vector store...")
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    return None

def save_vectorstore(vs: FAISS):
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(VECTORSTORE_DIR))

# ─────────────────────────────────────────────
# STAGE 1 — CRAWL
# ─────────────────────────────────────────────

def crawl_library():
    """
    Crawl giza.fas.harvard.edu/library and collect all PDF links + metadata.
    Saves result to manifest.json.
    """
    log.info(f"Starting crawl of {LIBRARY_URL}")
    session = requests.Session()
    session.headers.update({"User-Agent": "GizaResearchBot/1.0 (academic research)"})

    # ── Fetch index page ──
    resp = session.get(LIBRARY_URL, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Collect all internal document links (adjust selector if site structure changes)
    doc_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Library entries typically contain /library/ or /objects/
        if "/library/" in href or "/objects/" in href:
            full = href if href.startswith("http") else "http://giza.fas.harvard.edu" + href
            if full not in doc_links and full != LIBRARY_URL:
                doc_links.append(full)

    log.info(f"Found {len(doc_links)} document page candidates")

    manifest = []
    seen_pdfs = set()

    for idx, doc_url in enumerate(doc_links):
        try:
            log.info(f"[{idx+1}/{len(doc_links)}] Fetching metadata: {doc_url}")
            time.sleep(CRAWL_DELAY)

            r = session.get(doc_url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                log.warning(f"  Skipped (status {r.status_code})")
                continue

            dsoup = BeautifulSoup(r.text, "html.parser")

            # ── Extract PDF link ──
            pdf_url = None
            for a in dsoup.find_all("a", href=True):
                if a["href"].lower().endswith(".pdf"):
                    pdf_url = a["href"] if a["href"].startswith("http") \
                              else "http://giza.fas.harvard.edu" + a["href"]
                    break

            if not pdf_url or pdf_url in seen_pdfs:
                continue
            seen_pdfs.add(pdf_url)

            # ── Extract metadata (best-effort; adapt selectors to actual HTML) ──
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
            log.info(f"  → {title[:60]} ({year}) [{dtype}]")

        except Exception as e:
            log.error(f"  Error processing {doc_url}: {e}")

    save_json(MANIFEST_FILE, manifest)
    log.info(f"Manifest saved: {len(manifest)} PDFs — {MANIFEST_FILE}")
    return manifest


def _meta_or_text(soup, selectors: list) -> str:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el.get_text(separator=" ", strip=True)
    # Fallback: og/meta tags
    for prop in ["og:title", "DC.title", "citation_title"]:
        m = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if m and m.get("content"):
            return m["content"]
    return ""

def _extract_year(soup) -> str:
    text = soup.get_text()
    m = re.search(r'\b(1[89]\d{2}|20[012]\d)\b', text)
    return m.group(0) if m else "Unknown"

def _infer_doc_type(title: str, soup) -> str:
    t = title.lower()
    page_text = soup.get_text().lower()
    if any(w in t for w in ["report", "excavation", "season", "field"]):
        return "field_report"
    if any(w in t for w in ["journal", "article", "notes", "bulletin"]):
        return "article"
    if "dissertation" in t or "thesis" in t:
        return "thesis"
    if any(w in page_text for w in ["volume", "chapter", "isbn"]):
        return "book"
    return "publication"

# ─────────────────────────────────────────────
# STAGE 2 — EXTRACT
# ─────────────────────────────────────────────

def _is_page_background(width: int, height: int) -> bool:
    """
    Return True if this image is almost certainly a full-page background scan
    rather than a content illustration.
    Detects by aspect ratio matching standard page sizes within tolerance,
    combined with being large (>500k pixels — genuine content images are
    rarely this large AND page-shaped simultaneously).
    """
    if width == 0 or height == 0:
        return False
    ratio = width / height
    for page_ratio in _PAGE_RATIOS:
        if abs(ratio - page_ratio) < PAGE_RATIO_TOLERANCE:
            if width * height > 500_000:   # large AND page-shaped → background
                return True
    return False


def _extract_caption_near_image(page, img_xref: int) -> str:
    """
    Find a figure caption near an image on the page.
    Uses get_image_info(xrefs=True) to get correct bbox, with positional
    fallback for JPXDecode images where get_image_bbox() fails.
    """
    try:
        img_rect = None

        # Method 1: get_image_info with xrefs=True (required to populate xref field)
        for item in page.get_image_info(xrefs=True):
            if item.get("xref") == img_xref:
                bbox = item.get("bbox")
                if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    img_rect = fitz.Rect(bbox)
                break

        # Method 2: match by width/height from get_images() as positional fallback
        # (works when xref matching fails, e.g. JPXDecode via Form XObject)
        if img_rect is None:
            base_img_sizes = {}
            for img_info in page.get_images(full=True):
                xref, _, w, h = img_info[0], img_info[1], img_info[2], img_info[3]
                base_img_sizes[xref] = (w, h)

            target_size = base_img_sizes.get(img_xref)
            if target_size:
                for item in page.get_image_info(xrefs=True):
                    if item.get("width") == target_size[0] and item.get("height") == target_size[1]:
                        bbox = item.get("bbox")
                        if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            img_rect = fitz.Rect(bbox)
                        break

        if img_rect is None or img_rect.is_empty:
            return ""

        caption_pattern = re.compile(
            r'\b(fig\.?|figure|plate|pl\.?|photo|photograph|map|plan|table)\b',
            re.IGNORECASE
        )

        best_text = ""
        best_dist = float("inf")

        for block in page.get_text("blocks"):
            bx0, by0, bx1, by1 = block[0], block[1], block[2], block[3]
            text = block[4].strip()
            if not text or not caption_pattern.search(text):
                continue

            # Euclidean distance from nearest edge of image to nearest edge of text block
            # Note: caption may be inside or overlapping the image bbox, so clamp to 0
            dx = max(0.0, max(bx0 - img_rect.x1, img_rect.x0 - bx1))
            dy = max(0.0, max(by0 - img_rect.y1, img_rect.y0 - by1))
            dist = (dx**2 + dy**2) ** 0.5

            if dist < 200 and dist < best_dist:
                best_dist = dist
                best_text = text

        return best_text[:400] if best_text else ""
    except Exception:
        return ""
def extract_pdf(pdf_path: Path, doc_meta: dict) -> tuple[list[str], list[dict]]:
    """
    Extract text chunks, images, and tables from a PDF.
    Returns (text_chunks, image_records).
    image_records: list of {path, page, index, doc_id, caption, page_number}

    FIX: page-background images (full-page scans embedded as images) are
    detected by aspect ratio and discarded. Only genuine content images kept.
    """
    doc_id   = doc_meta.get("id", hashlib.md5(str(pdf_path).encode()).hexdigest()[:10])
    doc_name = _safe_name(doc_meta.get("title", pdf_path.stem))
    img_dir  = IMAGES_DIR / doc_name
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    full_text_pages = []
    image_records   = []
    bg_discarded    = 0
    tiny_discarded  = 0

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ── Text ──
        page_text = page.get_text("text")

        # ── Tables ──
        try:
            tables_on_page = page.find_tables()
            for t_idx, table in enumerate(tables_on_page):
                df = pd.DataFrame(table.extract())
                table_str = f"\n[TABLE — {doc_meta.get('title','?')}, p.{page_num+1}, table {t_idx+1}]\n"
                table_str += df.to_string(index=False, header=False) + "\n"
                page_text += table_str
        except Exception:
            pass

        # ── Images ──
        # Use a set to track xrefs already processed on this page
        # (same image can appear multiple times in get_images list)
        seen_xrefs = set()
        content_img_count = 0

        img_list = page.get_images(full=True)
        for img_info in img_list:
            try:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                base_img  = doc.extract_image(xref)
                img_bytes = base_img["image"]
                img_w     = base_img.get("width",  0)
                img_h     = base_img.get("height", 0)

                # ── Filter 1: discard tiny images (icons, bullets, decorative rules) ──
                if img_w * img_h < MIN_IMAGE_PIXELS:
                    tiny_discarded += 1
                    log.debug(f"  Discarded tiny image p{page_num+1} ({img_w}x{img_h})")
                    continue

                # ── Filter 2: discard full-page background scans ──
                if _is_page_background(img_w, img_h):
                    bg_discarded += 1
                    log.debug(f"  Discarded background image p{page_num+1} ({img_w}x{img_h})")
                    continue

                content_img_count += 1
                img_filename = f"page_{page_num+1:04d}_img_{content_img_count:03d}.png"
                img_path     = img_dir / img_filename

                # Always save as PNG via Pillow
                pil_img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                pil_img.save(img_path, "PNG")

                # ── Caption extraction (searches all four sides) ──
                caption = _extract_caption_near_image(page, xref)

                image_records.append({
                    "doc_id":    doc_id,
                    "doc_title": doc_meta.get("title", ""),
                    "page":      page_num + 1,
                    "img_index": content_img_count,
                    "path":      str(img_path),
                    "caption":   caption,
                })

                # Inject caption into page text so it gets embedded and distilled
                if caption:
                    page_text += f"\n[FIGURE CAPTION, p.{page_num+1}]: {caption}\n"

            except Exception as e:
                log.debug(f"  Image extract error p{page_num+1}: {e}")

        full_text_pages.append(page_text)

    doc.close()

    if bg_discarded or tiny_discarded:
        log.info(f"  Images discarded: {bg_discarded} backgrounds, {tiny_discarded} tiny")

    # ── Chunk ──
    full_text = "\n".join(full_text_pages)
    chunks    = _chunk_text(full_text)
    log.info(f"  Extracted {len(chunks)} chunks, {len(image_records)} content images from {pdf_path.name}")
    return chunks, image_records


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping word-based chunks, tagged with approximate
    page number so claims can later reference their source page.
    Each chunk is hard-capped at MAX_CHUNK_CHARS characters.
    """
    words  = text.split()
    step   = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
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
        log.warning(f"  {trimmed} chunks were trimmed to {MAX_CHUNK_CHARS} chars (context limit)")
    return chunks

def _safe_name(name: str) -> str:
    return re.sub(r'[^\w\-_]', '_', name)[:60]

# ─────────────────────────────────────────────
# STAGE 3 — DEDUPLICATE + EMBED
# ─────────────────────────────────────────────

def embed_chunks_with_deduplication(
    chunks:    list[str],
    doc_meta:  dict,
    image_refs: list[dict],
    vectorstore: Optional[FAISS],
    llm
) -> tuple[FAISS, int, int, int]:
    """
    Embed chunks, skipping duplicates.
    Returns (updated_vectorstore, added, skipped_dup, flagged_llm)
    """
    embeddings = get_embeddings()
    added = skipped = llm_checks = 0

    page_to_images = {}
    for rec in image_refs:
        page_to_images.setdefault(rec["page"], []).append(rec["path"])

    docs_to_add = []

    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_CHARS:
            chunk = chunk[:MAX_CHUNK_CHARS]

        if not vectorstore:
            docs_to_add.append(_make_doc(chunk, doc_meta))
            added += 1
            continue

        results = vectorstore.similarity_search_with_score(chunk, k=1)

        if not results:
            docs_to_add.append(_make_doc(chunk, doc_meta))
            added += 1
            continue

        best_doc, score = results[0]
        similarity = max(0.0, 1.0 - score / 2.0)

        if similarity >= SIMILARITY_DISCARD:
            skipped += 1
            log.debug(f"  SKIP (sim={similarity:.3f}): {chunk[:80]}…")

        elif similarity >= SIMILARITY_CHECK:
            llm_checks += 1
            is_dup = _llm_duplicate_check(chunk, best_doc.page_content, llm)
            if is_dup:
                skipped += 1
                log.debug(f"  LLM-SKIP (sim={similarity:.3f}): {chunk[:80]}…")
            else:
                docs_to_add.append(_make_doc(chunk, doc_meta))
                added += 1
        else:
            docs_to_add.append(_make_doc(chunk, doc_meta))
            added += 1

    if docs_to_add:
        if vectorstore:
            vectorstore.add_documents(docs_to_add)
        else:
            vectorstore = FAISS.from_documents(docs_to_add, embeddings)

    log.info(f"  Chunks: {added} added, {skipped} skipped, {llm_checks} LLM checks")
    return vectorstore, added, skipped, llm_checks


def _make_doc(chunk: str, meta: dict) -> Document:
    return Document(
        page_content=chunk,
        metadata={
            "doc_id":   meta.get("id", ""),
            "title":    meta.get("title", ""),
            "author":   meta.get("author", ""),
            "year":     meta.get("year", ""),
            "type":     meta.get("type", ""),
        }
    )


def _llm_duplicate_check(chunk_a: str, chunk_b: str, llm) -> bool:
    prompt = f"""Are the following two text passages conveying the same factual information?
Answer with a single word: YES or NO.

Passage A:
{chunk_a[:600]}

Passage B:
{chunk_b[:600]}

Answer:"""
    response  = llm.invoke(prompt)
    answer    = strip_think_tags(response.content).strip().upper()
    return answer.startswith("Y")

# ─────────────────────────────────────────────
# STAGE 4 — DISTIL
# ─────────────────────────────────────────────

DISTIL_PROMPT = """You are a research analyst specialising in Egyptology and ancient engineering.

The following text is extracted from:
Title:  {title}
Author: {author}
Year:   {year}
Type:   {doc_type}

TEXT:
{text}

Task: Identify up to {max_claims} distinct factual claims, measurements, or interpretations that THIS document contributes.

Rules:
- Express each claim in 2-4 sentences.
- The FIRST sentence must state the finding with any specific numbers, measurements, or dates.
- The SECOND sentence must provide context: what object, structure, tomb, or material is being described, and where it is located.
- If relevant, add a third sentence explaining the method used or the significance of the finding.
- If the text includes a figure caption (marked [FIGURE CAPTION]), incorporate what the figure shows into the relevant claim.
- EXCLUDE claims that merely cite or summarise previous work without adding new data.
- EXCLUDE vague generalities with no specific data.
- Focus on what a researcher would specifically credit this source for.

Example of good output:
"The north wall plaster measured 2.5-4.0 cm in the primary layer. This measurement was taken from the tomb of Kay at Giza, an Old Kingdom official whose tomb paintings were analysed as part of a 1998 conservation study. X-ray diffraction of the same plaster confirmed the presence of quartz, calcite, gypsum, and dolomite."

Return ONLY a numbered list. No preamble or commentary.
"""

SUMMARY_PROMPT = """You are a research analyst specialising in Egyptology and ancient engineering.

The following claims were distilled from:
Title:  {title}
Author: {author}
Year:   {year}

CLAIMS:
{claims_text}

Write a single short descriptive title (8 words maximum) that captures the primary subject matter of this document's contribution.
Then write 2-3 sentences summarising what this document specifically contributes to the field.

Format your response exactly like this:
TITLE: <your short title here>
SUMMARY: <your 2-3 sentence summary here>

No other text.
"""


def distil_document(chunks: list[str], doc_meta: dict, llm) -> list[dict]:
    """
    Ask the LLM to distil a document's chunks into distinct claims,
    then generate a short descriptive title and summary for the document.
    Returns list of claim dicts, each with a 'doc_summary' and 'doc_label' field.
    """
    combined = " ".join(chunks)
    words    = combined.split()
    sample   = " ".join(words[:8000])

    prompt = DISTIL_PROMPT.format(
        title      = doc_meta.get("title",  "Unknown"),
        author     = doc_meta.get("author", "Unknown"),
        year       = doc_meta.get("year",   "Unknown"),
        doc_type   = doc_meta.get("type",   "publication"),
        text       = sample,
        max_claims = DISTILL_MAX_CLAIMS,
    )

    response = llm.invoke(prompt)
    raw      = strip_think_tags(response.content)

    claims = []
    for line in raw.splitlines():
        line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
        if len(line) > 20:
            claims.append({
                "claim":       line,
                "doc_id":      doc_meta.get("id",     ""),
                "title":       doc_meta.get("title",  ""),
                "author":      doc_meta.get("author", ""),
                "year":        doc_meta.get("year",   ""),
                "type":        doc_meta.get("type",   ""),
                "pdf_url":     doc_meta.get("pdf_url",""),
                "doc_label":   "",    # filled in below
                "doc_summary": "",    # filled in below
                "extracted":   datetime.now().isoformat(timespec="seconds"),
            })

    log.info(f"  Distilled {len(claims)} claims from {doc_meta.get('title','?')[:50]}")

    # ── Generate short descriptive label + summary ──
    if claims:
        claims_text = "\n".join(f"- {c['claim']}" for c in claims[:10])
        sum_prompt  = SUMMARY_PROMPT.format(
            title       = doc_meta.get("title",  "Unknown"),
            author      = doc_meta.get("author", "Unknown"),
            year        = doc_meta.get("year",   "Unknown"),
            claims_text = claims_text,
        )
        sum_response = llm.invoke(sum_prompt)
        sum_raw      = strip_think_tags(sum_response.content)

        doc_label   = ""
        doc_summary = ""
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
# STAGE 5 — EXPORT
# ─────────────────────────────────────────────

def export_ideas():
    """
    Read all distilled claims from progress records and write ideas.csv
    """
    progress = load_json(PROGRESS_FILE, {})
    all_claims = []
    for doc_id, rec in progress.items():
        all_claims.extend(rec.get("claims", []))

    if not all_claims:
        log.warning("No claims found. Run the pipeline first.")
        return

    fieldnames = [
        "claim", "doc_label", "doc_summary",
        "doc_id", "title", "author", "year", "type", "pdf_url", "extracted"
    ]
    with open(IDEAS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_claims)

    log.info(f"Exported {len(all_claims)} claims → {IDEAS_CSV}")
    print(f"\n✓ Exported {len(all_claims)} claims to {IDEAS_CSV}")

# ─────────────────────────────────────────────
# DOCUMENT PROCESSING ORCHESTRATOR
# ─────────────────────────────────────────────

def process_one(doc_meta: dict, pdf_path: Path, vectorstore, llm) -> tuple[FAISS, list[dict], list[dict]]:
    """Full pipeline for a single document. Returns (updated_vectorstore, claims, image_refs)."""
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {doc_meta.get('title','?')}")
    log.info(f"  Author: {doc_meta.get('author','?')}  Year: {doc_meta.get('year','?')}")

    chunks, image_refs = extract_pdf(pdf_path, doc_meta)

    vectorstore, added, skipped, llm_checks = embed_chunks_with_deduplication(
        chunks, doc_meta, image_refs, vectorstore, llm
    )
    save_vectorstore(vectorstore)

    claims = distil_document(chunks, doc_meta, llm)

    return vectorstore, claims, image_refs


def download_pdf(url: str, dest_dir: Path, filename: str) -> Optional[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        log.info(f"  PDF already downloaded: {dest}")
        return dest
    try:
        log.info(f"  Downloading {url}")
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info(f"  Saved to {dest}")
        return dest
    except Exception as e:
        log.error(f"  Download failed: {e}")
        return None

# ─────────────────────────────────────────────
# CLI COMMANDS
# ─────────────────────────────────────────────

def cmd_crawl(_args):
    manifest = crawl_library()
    print(f"\n✓ Manifest built: {len(manifest)} PDFs → {MANIFEST_FILE}")


def cmd_run(args):
    manifest = load_json(MANIFEST_FILE, None)
    if manifest is None:
        print("No manifest found. Run: python giza_pipeline.py crawl")
        return

    progress         = load_json(PROGRESS_FILE, {})
    processed_titles = load_json(PROCESSED_TITLES, [])
    vectorstore      = load_vectorstore()
    llm              = get_llm()
    pdfs_dir         = DATA_DIR / "pdfs"
    pdfs_dir.mkdir(exist_ok=True)

    limit = getattr(args, "limit", None)
    done  = 0

    for entry in manifest:
        if limit and done >= limit:
            break

        doc_id = entry["id"]
        title  = entry["title"]

        if doc_id in progress:
            log.debug(f"Already processed: {title[:50]}")
            continue

        if title in processed_titles:
            log.info(f"Title already in knowledge base, skipping: {title[:50]}")
            progress[doc_id] = {"skipped_title": True}
            save_json(PROGRESS_FILE, progress)
            continue

        safe_filename = _safe_name(title) + ".pdf"
        pdf_path = download_pdf(entry["pdf_url"], pdfs_dir, safe_filename)
        if not pdf_path:
            progress[doc_id] = {"error": "download_failed"}
            save_json(PROGRESS_FILE, progress)
            continue

        time.sleep(CRAWL_DELAY)

        try:
            vectorstore, claims, image_refs = process_one(entry, pdf_path, vectorstore, llm)
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
            log.info(f"  ✓ Done ({done} this session)")

        except Exception as e:
            log.error(f"  Failed: {e}", exc_info=True)
            progress[doc_id] = {"error": str(e)}
            save_json(PROGRESS_FILE, progress)

    total_done = sum(1 for v in progress.values() if "claims" in v)
    print(f"\n✓ Session complete. {done} processed this run, {total_done} total in DB.")


def cmd_single(args):
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

    vectorstore = load_vectorstore()
    llm         = get_llm()

    vectorstore, claims, image_refs = process_one(doc_meta, pdf_path, vectorstore, llm)

    # Print label and summary
    if claims:
        print(f"\n{'='*60}")
        print(f"DOCUMENT LABEL:   {claims[0].get('doc_label','')}")
        print(f"DOCUMENT SUMMARY: {claims[0].get('doc_summary','')}")

    print(f"\n{'='*60}")
    print(f"DISTILLED CLAIMS ({len(claims)}):")
    print(f"{'='*60}")
    for i, c in enumerate(claims, 1):
        print(f"{i:2}. {c['claim']}")

    # Export claims CSV
    if claims:
        out_csv = DATA_DIR / f"ideas_{doc_meta['id']}.csv"
        fieldnames = [
            "claim", "doc_label", "doc_summary",
            "doc_id", "title", "author", "year", "type", "pdf_url", "extracted"
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(claims)
        print(f"\n✓ Claims exported → {out_csv}")

    # Export image index CSV with captions
    if image_refs:
        img_csv = DATA_DIR / f"images_{doc_meta['id']}.csv"
        img_fields = ["doc_id", "doc_title", "page", "img_index", "path", "caption"]
        with open(img_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=img_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(image_refs)
        print(f"✓ Image index exported → {img_csv}")

        captioned = [r for r in image_refs if r.get("caption")]
        print(f"\nContent images saved: {len(image_refs)}")
        if captioned:
            print(f"Captions found ({len(captioned)}/{len(image_refs)}):")
            for r in captioned:
                print(f"  p.{r['page']} img {r['img_index']}: {r['caption'][:120]}")
        else:
            print("No figure captions detected (captions may use non-standard labels).")


def cmd_export(_args):
    export_ideas()


def cmd_purge(_args):
    import shutil
    confirm = input("This will delete the vector store, progress, and processed titles. Type YES to confirm: ")
    if confirm.strip().upper() != "YES":
        print("Cancelled.")
        return
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
        print(f"Deleted {VECTORSTORE_DIR}")
    for f in [PROGRESS_FILE, PROCESSED_TITLES]:
        if f.exists():
            f.unlink()
            print(f"Deleted {f}")
    print("✓ Database purged. Ready for a clean run.")


def cmd_status(_args):
    manifest  = load_json(MANIFEST_FILE,  [])
    progress  = load_json(PROGRESS_FILE,  {})
    titles    = load_json(PROCESSED_TITLES, [])

    total        = len(manifest)
    completed    = sum(1 for v in progress.values() if "claims"        in v)
    errors       = sum(1 for v in progress.values() if "error"         in v)
    skipped      = sum(1 for v in progress.values() if "skipped_title" in v)
    remaining    = total - len(progress)
    total_claims = sum(len(v.get("claims", [])) for v in progress.values())

    print(f"\n{'='*40}")
    print(f"  GIZA PIPELINE STATUS")
    print(f"{'='*40}")
    print(f"  Manifest PDFs:    {total}")
    print(f"  Completed:        {completed}")
    print(f"  Errors:           {errors}")
    print(f"  Skipped (title):  {skipped}")
    print(f"  Remaining:        {remaining}")
    print(f"  Total claims:     {total_claims}")
    print(f"  Unique titles:    {len(titles)}")
    print(f"{'='*40}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Giza Digital Library RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("crawl",  help="Stage 1: crawl library and build manifest")

    p_run = sub.add_parser("run", help="Stages 2-4: process PDFs")
    p_run.add_argument("--limit", type=int, default=None,
                       help="Process at most N unprocessed documents this session")

    p_single = sub.add_parser("single", help="Test pipeline on a local PDF")
    p_single.add_argument("path",              help="Absolute path to PDF file")
    p_single.add_argument("--title",  default=None)
    p_single.add_argument("--author", default=None)
    p_single.add_argument("--year",   default=None)
    p_single.add_argument("--type",   default=None,
                          help="publication / book / article / field_report / thesis")

    sub.add_parser("export", help="Stage 5: export all ideas to CSV")
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