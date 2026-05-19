"""
Run all quality diagnostics on the claims store.
Usage: python diagnose.py
"""
import json
import re
import random
from pathlib import Path
from collections import Counter

random.seed(123)

store = json.load(open('pipeline_data/claims_store.json', encoding='utf-8'))
idx   = json.load(open('pipeline_data/progress_index.json', encoding='utf-8'))

all_claims = []
for k, v in store.items():
    tier = idx.get(k, {}).get('tier', 1)
    for c in v.get('claims', []):
        all_claims.append((tier, c))

print(f"{'='*60}")
print("TEST 1 — RANDOM SAMPLE OF 20 FACTUAL CLAIMS")
print(f"{'='*60}")
sample = random.sample(all_claims, min(20, len(all_claims)))
for i, (tier, c) in enumerate(sample, 1):
    print(f"[{i}] Tier {tier} | {c.get('title','')[:50]}")
    print(c.get('claim',''))
    print()

print(f"{'='*60}")
print("TEST 2 — MEASUREMENT AND NUMBER CONTENT")
print(f"{'='*60}")
claims_only = [c for _, c in all_claims]
n = len(claims_only)

has_number  = sum(1 for c in claims_only if re.search(r'\d+\.?\d*', c['claim']))
has_unit    = sum(1 for c in claims_only if re.search(
    r'\d+\.?\d*\s*(mm|cm|m\b|km|inch|inches|feet|foot|ft|cubit|cubits|degree|degrees|hz|Hz|kg|ton|percent|%)',
    c['claim'], re.I))
has_decimal = sum(1 for c in claims_only if re.search(r'\d+\.\d+', c['claim']))
has_date    = sum(1 for c in claims_only if re.search(
    r'\b(BC|BCE|AD|CE|\d{3,4}\s*(BC|BCE|AD|CE))', c['claim']))
has_angle   = sum(1 for c in claims_only if re.search(
    r'\d+\s*(degree|degrees|°)', c['claim'], re.I))

print(f"Total claims            : {n}")
print(f"Contains any number     : {has_number:5d} ({has_number*100//n}%)")
print(f"Contains measurement+unit: {has_unit:5d} ({has_unit*100//n}%)")
print(f"Contains decimal number : {has_decimal:5d} ({has_decimal*100//n}%)")
print(f"Contains BC/AD date     : {has_date:5d} ({has_date*100//n}%)")
print(f"Contains angle/degrees  : {has_angle:5d} ({has_angle*100//n}%)")
print()

with_units = [c for c in claims_only if re.search(
    r'\d+\.?\d*\s*(mm|cm|m\b|km|inch|inches|feet|foot|ft|cubit|cubits|degree|degrees|hz|Hz|kg|ton)',
    c['claim'], re.I)]
print(f"--- 10 claims WITH measurement units ({len(with_units)} total) ---")
random.seed(1)
for c in random.sample(with_units, min(10, len(with_units))):
    print(c['claim'][:250])
    print()

no_numbers = [c for c in claims_only if not re.search(r'\d', c['claim'])]
print(f"--- Claims with NO numbers at all: {len(no_numbers)} ({len(no_numbers)*100//n}%) ---")
print("Sample of 10:")
for c in random.sample(no_numbers, min(10, len(no_numbers))):
    print(f"  [{c.get('title','')[:40]}]")
    print(f"  {c['claim'][:200]}")
    print()

print(f"{'='*60}")
print("TEST 3 — NON-ASCII / ENCODING ARTIFACTS")
print(f"{'='*60}")
garbled = [c for c in claims_only if re.search(r'[^\x00-\x7F]', c['claim'])]
print(f"Claims with non-ASCII characters: {len(garbled)}")
print()
print("Sample 10 non-ASCII claims:")
random.seed(5)
for c in random.sample(garbled, min(10, len(garbled))):
    print(f"  [{c.get('title','')[:40]}]")
    print(f"  {c['claim'][:300]}")
    print()

print(f"{'='*60}")
print("TEST 4 — PETRIE DOCUMENTS")
print(f"{'='*60}")
petrie_docs = [(k, v) for k, v in store.items()
               if 'petrie' in idx.get(k, {}).get('title', '').lower()]
print(f"Petrie documents found: {len(petrie_docs)}")
print()
for doc_id, rec in sorted(petrie_docs, key=lambda x: len(x[1].get('claims', [])), reverse=True)[:5]:
    title  = idx.get(doc_id, {}).get('title', '?')
    claims = rec.get('claims', [])
    print(f"{title} — {len(claims)} claims")
    print("First 5 claims:")
    for c in claims[:5]:
        print(f"  {c['claim'][:300]}")
    print()

print(f"{'='*60}")
print("TEST 5 — PETRIE PYRAMIDS AND TEMPLES OF GIZEH (all claims)")
print(f"{'='*60}")
target = None
for k, v in store.items():
    title = idx.get(k, {}).get('title', '')
    tl = title.lower()
    if ('pyramids' in tl and 'temples' in tl and 'gizeh' in tl) or \
       ('petrie' in tl and 'gizeh' in tl):
        target = (k, v, title)
        break

if target:
    doc_id, rec, title = target
    claims = rec.get('claims', [])
    print(f"Document: {title}")
    print(f"Total claims: {len(claims)}")
    print()
    for i, c in enumerate(claims, 1):
        print(f"{i:3d}. {c['claim']}")
        print()
else:
    print("Not found. Titles containing 'petrie':")
    for k in store:
        t = idx.get(k, {}).get('title', '')
        if 'petrie' in t.lower():
            print(f"  {t} ({len(store[k].get('claims',[]))} claims)")

print(f"{'='*60}")
print("TEST 6 — AUTHOR ATTRIBUTION QUALITY")
print(f"{'='*60}")
unknown_author = sum(1 for c in claims_only
                     if c.get('author', '').strip().lower() in ('unknown', '', 'none'))
known_author   = n - unknown_author
print(f"Total claims    : {n}")
print(f"Known author    : {known_author} ({known_author*100//n}%)")
print(f"Unknown author  : {unknown_author} ({unknown_author*100//n}%)")
print()
authors = Counter(c.get('author', 'Unknown') for c in claims_only
                  if c.get('author', '').strip().lower() not in ('unknown', '', 'none'))
print("Top 20 authors by claim count:")
for author, count in authors.most_common(20):
    print(f"  {count:5d}  {author[:60]}")

print(f"{'='*60}")
print("TEST 7 — FUNCTIONAL CLAIMS SAMPLE")
print(f"{'='*60}")
all_func = []
for k, v in store.items():
    tier = idx.get(k, {}).get('tier', 1)
    for c in v.get('functional_claims', []):
        all_func.append((tier, c.get('title', ''), c.get('claim', '')))

print(f"Total functional claims: {len(all_func)}")
print()
random.seed(42)
sample_func = random.sample(all_func, min(20, len(all_func)))
for i, (tier, title, claim) in enumerate(sample_func, 1):
    print(f"[{i}] Tier {tier} | {title[:50]}")
    print(claim)
    print()

print(f"{'='*60}")
print("TEST 8 — VAGUE FUNCTIONAL CLAIM DETECTION")
print(f"{'='*60}")
vague_patterns = [
    r'according to unknown',
    r'the text notes the theory',
    r'may have (been|served|involved|had)',
    r'possibly (used|served|designed)',
    r'could have',
    r'some (speculative|theories|researchers)',
    r'were (designed|used|built) (to|for) [a-z]+ (purposes|functions|reasons)',
]
vague_count = 0
for tier, title, claim in all_func:
    for pat in vague_patterns:
        if re.search(pat, claim, re.I):
            vague_count += 1
            break

print(f"Functional claims matching vague patterns: {vague_count} ({vague_count*100//max(len(all_func),1)}%)")
print()
print("Sample vague functional claims:")
vague = [(t, ti, c) for t, ti, c in all_func
         if any(re.search(p, c, re.I) for p in vague_patterns)]
random.seed(7)
for tier, title, claim in random.sample(vague, min(10, len(vague))):
    print(f"  [Tier {tier}] {title[:50]}")
    print(f"  {claim[:200]}")
    print()