"""
data_cleaning.py
================
Utilities for preparing PDFs before pipeline ingestion.

Commands:
  python data_cleaning.py ocr <file.pdf>              OCR a single image-only PDF
  python data_cleaning.py ocr <file.pdf> --lang deu   OCR with specific language (deu, fra, ita)
  python data_cleaning.py ocr-folder <folder>         OCR all image-only PDFs in a folder
  python data_cleaning.py check <file.pdf>            Check if a PDF needs OCR or translation
  python data_cleaning.py check-folder <folder>       Check all PDFs in a folder
  python data_cleaning.py translate <file.pdf>        Translate a PDF to English
  python data_cleaning.py translate-folder <folder>   Translate all non-English PDFs in a folder
  python data_cleaning.py scan <folder>               Full scan: report image-only AND foreign language PDFs
  python data_cleaning.py process-folder <folder>     Full auto pipeline: OCR + translate + rename + archive

The process-folder command does everything automatically:
  1. Scans every PDF in the folder
  2. If image-only: runs OCR, deletes .bak on success
  3. Re-checks text after OCR
  4. If foreign language: translates to English, renames with translated title
     e.g. "Cantu Gianni - Le Mystere Des Pyramides.pdf"
      ->  "Cantu Gianni - The Mystery of the Pyramids.pdf"
  5. Moves the original file to master_copies before replacing it
  6. Leaves only clean English text PDFs in the source folder

Examples:
  python data_cleaning.py ocr "tier_3/Petrie W.M.F. - Inductive Metrology.pdf" --lang eng
  python data_cleaning.py ocr "tier_2/Borchardt Ludwig - Das Grabdenkmal.pdf" --lang deu
  python data_cleaning.py translate "tier_4/Cantu Gianni - Le Mystere Des Pyramides.pdf"
  python data_cleaning.py scan tier_2
  python data_cleaning.py process-folder tier_4
  python data_cleaning.py process-folder tier_4 --master "C:/Users/Ant/Documents/projects/knowledge_distillation_pipeline/master_copies"

Notes:
  - OCR requires: pip install ocrmypdf
  - OCR also requires Tesseract installed on Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - Translation requires: pip install deep-translator langdetect reportlab
  - Language codes: deu (German), fra (French), ita (Italian), ara (Arabic),
                    jpn (Japanese), nld (Dutch), spa (Spanish), por (Portuguese)
"""

import os
import sys
import re
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path


DEFAULT_MASTER = Path("master_copies")

LANG_MAP = {
    "german": "deu", "deutsch": "deu", "de": "deu", "deu": "deu",
    "french": "fra", "francais": "fra", "fr": "fra", "fra": "fra",
    "italian": "ita", "italiano": "ita", "it": "ita", "ita": "ita",
    "arabic": "ara", "ar": "ara", "ara": "ara",
    "japanese": "jpn", "ja": "jpn", "jpn": "jpn",
    "dutch": "nld", "nl": "nld", "nld": "nld",
    "spanish": "spa", "espanol": "spa", "es": "spa", "spa": "spa",
    "portuguese": "por", "pt": "por", "por": "por",
    "english": "eng", "en": "eng", "eng": "eng",
}

LANGDETECT_TO_TESSERACT = {
    "de": "deu", "fr": "fra", "it": "ita", "ar": "ara",
    "ja": "jpn", "nl": "nld", "es": "spa", "pt": "por",
    "en": "eng", "la": "lat",
}

FOREIGN_LANGUAGES = {
    "de": "German", "fr": "French", "it": "Italian",
    "ar": "Arabic", "ja": "Japanese", "nl": "Dutch",
    "es": "Spanish", "pt": "Portuguese", "la": "Latin",
    "ru": "Russian", "pl": "Polish", "cs": "Czech",
}


# ─────────────────────────────────────────────
# TEXT DENSITY + LANGUAGE DETECTION
# ─────────────────────────────────────────────

def get_text_density(pdf_path: Path) -> tuple:
    """Returns (total_pages, pages_with_text, ratio)."""
    try:
        import pymupdf as fitz
    except ImportError:
        print("ERROR: pymupdf not installed. Run: pip install pymupdf")
        sys.exit(1)

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    with_text = sum(1 for p in doc if len(p.get_text("text").strip()) > 50)
    doc.close()
    ratio = with_text / total if total > 0 else 0.0
    return total, with_text, ratio


def is_image_only(pdf_path: Path, threshold: float = 0.1) -> bool:
    _, _, ratio = get_text_density(pdf_path)
    return ratio < threshold


def detect_language(pdf_path: Path, sample_pages: int = 8) -> str:
    """Sample text from up to sample_pages pages and detect language."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        import pymupdf as fitz
    except ImportError:
        return "unknown"

    doc   = fitz.open(str(pdf_path))
    texts = []
    for i, page in enumerate(doc):
        if i >= sample_pages:
            break
        text = page.get_text("text").strip()
        if len(text) > 100:
            texts.append(text[:800])
    doc.close()

    if not texts:
        return "unknown"
    try:
        return detect(" ".join(texts))
    except Exception:
        return "unknown"


def _guess_lang_from_filename(stem: str) -> str:
    """
    Guess language from the title portion of the filename.
    Uses the part after ' - ' if present, otherwise the whole stem.
    Title words are a far more reliable signal than author surnames.
    """
    # Use title part only if author separator present
    if " - " in stem:
        title_part = stem.split(" - ", 1)[1].lower()
    else:
        title_part = stem.lower()

    # German: common words and endings in titles
    german_title_signals = [
        "das ", "die ", "der ", "des ", "dem ", "den ",
        "und ", "von ", "zum ", "zur ", "bei ", "mit ",
        "grabdenkmal", "konigs", "pyramide", "agypten",
        "aegypten", "untersuchung", "geschichte", "kunst",
        "bauperiode", "tempel", "archaologie",
    ]
    # French: common words in titles
    french_title_signals = [
        "le ", "la ", "les ", "des ", "du ", "de la ", "de l",
        "un ", "une ", "sur ", "au ", "aux ",
        "egypte", "pyramides", "mystere", "recherches",
        "observations", "fouilles", "monuments", "antiquites",
        "decouverte", "voyage", "histoire",
    ]
    # Italian: common words in titles
    italian_title_signals = [
        "il ", "lo ", "la ", "le ", "gli ", "i ",
        "della ", "delle ", "degli ", "del ",
        "piramidi", "egitto", "architettura",
        "scavi", "museo",
    ]

    for sig in german_title_signals:
        if sig in title_part:
            return "deu"
    for sig in french_title_signals:
        if sig in title_part:
            return "fra"
    for sig in italian_title_signals:
        if sig in title_part:
            return "ita"

    return "eng"


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

def check_ocrmypdf() -> bool:
    result = subprocess.run(["ocrmypdf", "--version"], capture_output=True, text=True)
    return result.returncode == 0


def run_ocr(pdf_path: Path, lang: str = "eng", force: bool = False,
            keep_bak: bool = True) -> bool:
    """
    OCR a PDF in-place. Backs up the original as .bak first.
    keep_bak=False deletes the backup on success.
    Returns True on success.
    """
    if not check_ocrmypdf():
        print("  ERROR: ocrmypdf not found. Install: pip install ocrmypdf")
        print("  Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

    lang_code = LANG_MAP.get(lang.lower(), lang)

    if not force:
        _, _, ratio = get_text_density(pdf_path)
        if ratio > 0.5:
            print(f"  Already {ratio*100:.0f}% text — skipping OCR (use --force to override)")
            return True

    backup_path = pdf_path.with_suffix(".bak")
    if not backup_path.exists():
        shutil.copy2(pdf_path, backup_path)
        print(f"  Backup: {backup_path.name}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = ["ocrmypdf", "--language", lang_code, "--output-type", "pdf",
               "--optimize", "0", "--skip-text", str(pdf_path), str(tmp_path)]
        if force:
            cmd.remove("--skip-text")
            cmd.insert(-2, "--force-ocr")

        print(f"  OCR ({lang_code}): {pdf_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            shutil.move(str(tmp_path), str(pdf_path))
            _, with_text, ratio = get_text_density(pdf_path)
            print(f"  OCR done: {ratio*100:.0f}% pages now have text")

            if not keep_bak and backup_path.exists():
                backup_path.unlink()
                print(f"  Backup removed")
            return True
        else:
            print(f"  OCR FAILED: {result.stderr[-400:]}")
            tmp_path.unlink(missing_ok=True)
            if backup_path.exists():
                shutil.copy2(backup_path, pdf_path)
                backup_path.unlink()
                print(f"  Restored from backup")
            return False

    except Exception as e:
        print(f"  OCR error: {e}")
        tmp_path.unlink(missing_ok=True)
        return False


# ─────────────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────────────

def check_translation_deps() -> bool:
    try:
        import deep_translator
        import langdetect
        return True
    except ImportError:
        print("ERROR: pip install deep-translator langdetect reportlab")
        return False


def translate_title(title_part: str) -> str:
    """Translate just the title portion to English for use in the filename."""
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="en").translate(title_part.strip())
        if translated:
            translated = translated.title()
            translated = re.sub(r'[<>:"/\\|?*]', '', translated).strip()
            return translated
        return title_part
    except Exception as e:
        print(f"  Title translation failed: {e}")
        return title_part


def translate_pdf(pdf_path: Path, output_path: Path = None) -> "Path | None":
    """
    Translate PDF content to English.
    Saves to output_path if given, otherwise <stem>_EN.pdf.
    Returns the output path on success, None on failure.
    """
    if not check_translation_deps():
        return None

    try:
        import pymupdf as fitz
        from deep_translator import GoogleTranslator
    except ImportError as e:
        print(f"  ERROR: {e}")
        return None

    dest = output_path or pdf_path.with_name(pdf_path.stem + "_EN.pdf")

    if dest.exists():
        print(f"  Already exists: {dest.name} — skipping")
        return dest

    lang = detect_language(pdf_path)
    if lang not in FOREIGN_LANGUAGES and lang not in ("unknown",):
        print(f"  Detected as English — skipping translation")
        return pdf_path

    lang_name = FOREIGN_LANGUAGES.get(lang, lang)
    print(f"  Translating {lang_name} → English: {pdf_path.name}")

    doc        = fitz.open(str(pdf_path))
    translator = GoogleTranslator(source="auto", target="en")
    pages_text = []

    for n, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            pages_text.append("")
            continue
        chunks     = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated = []
        for chunk in chunks:
            try:
                translated.append(translator.translate(chunk) or chunk)
            except Exception:
                translated.append(chunk)
        pages_text.append(" ".join(translated))
        if (n + 1) % 20 == 0:
            print(f"    Page {n+1}/{len(doc)}...")

    doc.close()

    # Write translated PDF
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                         Spacer, PageBreak)
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        styles = getSampleStyleSheet()
        story  = []
        for text in pages_text:
            if text:
                for para in text.split("\n"):
                    para = para.strip()
                    if para:
                        try:
                            story.append(Paragraph(para, styles["Normal"]))
                            story.append(Spacer(1, 4))
                        except Exception:
                            pass
            story.append(PageBreak())

        SimpleDocTemplate(
            str(dest),
            pagesize=A4,
            rightMargin=inch, leftMargin=inch,
            topMargin=inch,   bottomMargin=inch,
        ).build(story)
        print(f"  Saved: {dest.name}")
        return dest

    except ImportError:
        # Fallback: plain text file
        txt = dest.with_suffix(".txt")
        with open(txt, "w", encoding="utf-8") as f:
            for i, text in enumerate(pages_text):
                f.write(f"\n{'='*60}\nPAGE {i+1}\n{'='*60}\n{text}\n")
        print(f"  reportlab not installed — saved as text: {txt.name}")
        print(f"  Install: pip install reportlab")
        return txt

    except Exception as e:
        print(f"  Write error: {e}")
        return None


# ─────────────────────────────────────────────
# PROCESS FOLDER  — THE MAIN AUTO PIPELINE
# ─────────────────────────────────────────────

def process_folder(folder: Path, master_dir: Path) -> None:
    """
    Full automated cleaning pipeline for a folder of PDFs.

    For each PDF:
      1. If image-only  → OCR it, delete .bak on success
      2. Re-check after OCR
      3. If foreign     → translate title + content, save with English name,
                          archive original to master_dir, remove original from folder
      4. If already OK  → leave untouched
    """
    master_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in folder.rglob("*.pdf")
                  if not p.stem.endswith("_EN") and p.suffix != ".bak")

    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    print(f"Processing {len(pdfs)} PDFs in {folder}")
    print(f"Originals archived to: {master_dir}")
    print()

    stats = {"clean": 0, "ocr": 0, "translated": 0, "errors": 0}

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")

        try:
            # ── 1. OCR if image-only ───────────────────────────────
            _, _, ratio = get_text_density(pdf)

            if ratio < 0.1:
                print(f"  Image-only ({ratio*100:.0f}% text)")
                ocr_lang = _guess_lang_from_filename(pdf.stem)
                print(f"  Language guess from filename: {ocr_lang}")

                ok = run_ocr(pdf, lang=ocr_lang, keep_bak=False)
                if not ok:
                    print(f"  OCR failed — skipping")
                    stats["errors"] += 1
                    print()
                    continue

                stats["ocr"] += 1
                _, _, ratio = get_text_density(pdf)
                print(f"  Post-OCR text coverage: {ratio*100:.0f}%")

                if ratio < 0.1:
                    print(f"  Still no text after OCR — skipping")
                    stats["errors"] += 1
                    print()
                    continue

            # ── 2. Translate if foreign language ──────────────────
            lang = detect_language(pdf)
            print(f"  Language: {lang} ({FOREIGN_LANGUAGES.get(lang, 'English')})")

            if lang in FOREIGN_LANGUAGES:
                stem = pdf.stem

                # Build new filename with translated title
                if " - " in stem:
                    author_part, title_part = stem.split(" - ", 1)
                    print(f"  Translating title: '{title_part}'")
                    eng_title = translate_title(title_part)
                    print(f"  English title:     '{eng_title}'")
                    new_stem = f"{author_part} - {eng_title}"
                else:
                    eng_title = translate_title(stem)
                    new_stem  = eng_title

                new_pdf = pdf.parent / f"{new_stem}.pdf"
                # Avoid collision
                if new_pdf.exists() and new_pdf.resolve() != pdf.resolve():
                    new_pdf = pdf.parent / f"{new_stem}_tr.pdf"

                # Archive original
                archive_dest = master_dir / pdf.name
                if not archive_dest.exists():
                    shutil.copy2(pdf, archive_dest)
                    print(f"  Archived original → {archive_dest}")

                # Translate content → new path
                result = translate_pdf(pdf, output_path=new_pdf)

                if result and result.exists():
                    pdf.unlink()
                    print(f"  Original removed from tier folder")
                    print(f"  Clean English PDF: {new_pdf.name}")
                    stats["translated"] += 1
                else:
                    print(f"  Translation failed — original left in place")
                    # Remove archive copy since process didn't complete
                    if archive_dest.exists():
                        archive_dest.unlink()
                    stats["errors"] += 1

            else:
                print(f"  Already English — no action needed")
                stats["clean"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["errors"] += 1

        print()

    print(f"{'='*60}")
    print(f"COMPLETE: {folder}")
    print(f"{'='*60}")
    print(f"  Already clean : {stats['clean']}")
    print(f"  OCR'd         : {stats['ocr']}")
    print(f"  Translated    : {stats['translated']}")
    print(f"  Errors        : {stats['errors']}")
    print(f"  Originals at  : {master_dir}")


# ─────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────

def scan_folder(folder: Path) -> None:
    pdfs = sorted(p for p in folder.rglob("*.pdf")
                  if not p.stem.endswith("_EN") and p.suffix != ".bak")

    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    print(f"Scanning {len(pdfs)} PDFs in {folder}...")
    print()

    image_only_list   = []
    foreign_lang_list = []
    clean_list        = []
    errors_list       = []

    for i, pdf in enumerate(pdfs, 1):
        try:
            total, with_text, ratio = get_text_density(pdf)
            print(f"  [{i:3d}/{len(pdfs)}] {pdf.name[:60]}", end="", flush=True)

            if ratio < 0.1:
                image_only_list.append((pdf, total, ratio))
                print(f" → IMAGE-ONLY ({total} pages)")
            else:
                lang = detect_language(pdf)
                if lang in FOREIGN_LANGUAGES:
                    foreign_lang_list.append((pdf, lang, FOREIGN_LANGUAGES[lang], ratio))
                    print(f" → {FOREIGN_LANGUAGES[lang].upper()} ({ratio*100:.0f}% text)")
                else:
                    clean_list.append(pdf)
                    print(f" → OK ({ratio*100:.0f}% text, lang={lang})")
        except Exception as e:
            errors_list.append((pdf, str(e)))
            print(f" → ERROR: {e}")

    print()
    print(f"{'='*60}")
    print(f"SCAN RESULTS: {folder}")
    print(f"{'='*60}")
    print(f"  Total        : {len(pdfs)}")
    print(f"  Clean/English: {len(clean_list)}")
    print(f"  Image-only   : {len(image_only_list)}  ← need OCR")
    print(f"  Foreign lang : {len(foreign_lang_list)}  ← need translation")
    print(f"  Errors       : {len(errors_list)}")
    print()

    if image_only_list:
        print("─── IMAGE-ONLY ───")
        for pdf, pages, _ in sorted(image_only_list, key=lambda x: x[0].name):
            print(f"  {pdf.name}  ({pages} pages)")
        print()

    if foreign_lang_list:
        print("─── FOREIGN LANGUAGE ───")
        groups: dict = {}
        for pdf, lang, name, _ in foreign_lang_list:
            groups.setdefault(name, []).append(pdf)
        for name, files in sorted(groups.items()):
            print(f"  {name}:")
            for f in files:
                print(f"    {f.name}")
        print()
        print(f"To process everything automatically:")
        print(f"  python data_cleaning.py process-folder {folder}")

    if errors_list:
        print("─── ERRORS ───")
        for pdf, err in errors_list:
            print(f"  {pdf.name}: {err}")


# ─────────────────────────────────────────────
# CHECK SINGLE FILE
# ─────────────────────────────────────────────

def check_file(pdf_path: Path) -> None:
    total, with_text, ratio = get_text_density(pdf_path)
    lang      = detect_language(pdf_path)
    lang_name = FOREIGN_LANGUAGES.get(lang, lang)

    print(f"File    : {pdf_path.name}")
    print(f"Pages   : {total}")
    print(f"Text    : {with_text} pages ({ratio*100:.0f}%)")
    print(f"Language: {lang} ({lang_name})")
    print()

    if ratio < 0.1:
        lang_code = LANGDETECT_TO_TESSERACT.get(lang, "eng")
        print(f"STATUS: IMAGE-ONLY")
        print(f"  Run: python data_cleaning.py ocr \"{pdf_path}\" --lang {lang_code}")
    elif lang in FOREIGN_LANGUAGES:
        print(f"STATUS: FOREIGN ({lang_name})")
        print(f"  Run: python data_cleaning.py translate \"{pdf_path}\"")
    else:
        print("STATUS: CLEAN — ready for pipeline")


# ─────────────────────────────────────────────
# OCR FOLDER
# ─────────────────────────────────────────────

def ocr_folder(folder: Path, lang: str = None, force: bool = False) -> None:
    pdfs = [p for p in sorted(folder.rglob("*.pdf"))
            if not p.stem.endswith("_EN") and p.suffix != ".bak"]

    image_pdfs = []
    for pdf in pdfs:
        try:
            _, _, ratio = get_text_density(pdf)
            if ratio < 0.1 or force:
                image_pdfs.append(pdf)
        except Exception as e:
            print(f"  Error checking {pdf.name}: {e}")

    print(f"Found {len(image_pdfs)} image-only PDFs")
    success = failed = 0

    for i, pdf in enumerate(image_pdfs, 1):
        print(f"[{i}/{len(image_pdfs)}] {pdf.name}")
        ocr_lang = lang or LANGDETECT_TO_TESSERACT.get(detect_language(pdf), None) \
                         or _guess_lang_from_filename(pdf.stem)
        print(f"  Language: {ocr_lang}")
        if run_ocr(pdf, lang=ocr_lang, force=force, keep_bak=True):
            success += 1
        else:
            failed += 1
        print()

    print(f"OCR complete: {success} succeeded, {failed} failed")


# ─────────────────────────────────────────────
# TRANSLATE FOLDER
# ─────────────────────────────────────────────

def translate_folder(folder: Path) -> None:
    pdfs = [p for p in sorted(folder.rglob("*.pdf"))
            if not p.stem.endswith("_EN")]

    foreign = [(p, detect_language(p)) for p in pdfs]
    foreign = [(p, l) for p, l in foreign if l in FOREIGN_LANGUAGES]

    print(f"Found {len(foreign)} foreign language PDFs")
    for i, (pdf, lang) in enumerate(foreign, 1):
        print(f"[{i}/{len(foreign)}] {pdf.name} ({FOREIGN_LANGUAGES[lang]})")
        translate_pdf(pdf)
        print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PDF data cleaning utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_ocr = sub.add_parser("ocr", help="OCR a single image-only PDF")
    p_ocr.add_argument("path")
    p_ocr.add_argument("--lang", default=None)
    p_ocr.add_argument("--force", action="store_true")

    p_ocrf = sub.add_parser("ocr-folder", help="OCR all image-only PDFs in a folder")
    p_ocrf.add_argument("folder")
    p_ocrf.add_argument("--lang", default=None)
    p_ocrf.add_argument("--force", action="store_true")

    p_check = sub.add_parser("check", help="Check a single PDF")
    p_check.add_argument("path")

    p_cf = sub.add_parser("check-folder", help="Check all PDFs in a folder")
    p_cf.add_argument("folder")

    p_tr = sub.add_parser("translate", help="Translate a PDF to English")
    p_tr.add_argument("path")

    p_trf = sub.add_parser("translate-folder", help="Translate all foreign PDFs")
    p_trf.add_argument("folder")

    p_scan = sub.add_parser("scan", help="Scan folder and report issues")
    p_scan.add_argument("folder")

    p_proc = sub.add_parser(
        "process-folder",
        help="Full auto pipeline: OCR + translate + rename + archive"
    )
    p_proc.add_argument("folder", help="Folder to process")
    p_proc.add_argument("--master", default=None,
                        help="Master copies folder (default: ./master_copies)")

    args = parser.parse_args()

    if args.command == "ocr":
        p = Path(args.path)
        if not p.exists():
            print(f"Not found: {p}")
            sys.exit(1)
        run_ocr(p, lang=args.lang or "eng", force=args.force, keep_bak=True)

    elif args.command == "ocr-folder":
        ocr_folder(Path(args.folder), lang=args.lang, force=args.force)

    elif args.command == "check":
        p = Path(args.path)
        if not p.exists():
            print(f"Not found: {p}")
            sys.exit(1)
        check_file(p)

    elif args.command == "check-folder":
        scan_folder(Path(args.folder))

    elif args.command == "translate":
        p = Path(args.path)
        if not p.exists():
            print(f"Not found: {p}")
            sys.exit(1)
        translate_pdf(p)

    elif args.command == "translate-folder":
        translate_folder(Path(args.folder))

    elif args.command == "scan":
        scan_folder(Path(args.folder))

    elif args.command == "process-folder":
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            sys.exit(1)
        master = Path(args.master) if args.master else DEFAULT_MASTER
        process_folder(folder, master)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
