#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Zephyr‑style RAG prompts from pre‑2024 CVE data (CVE‑List V5).

Steps
-----
1.  Download or `git clone` https://github.com/CVEProject/cvelistV5
    ├─ zip it as  cve_data.zip   …or…
    └─ extract to ./cve_data/
2.  pip install numpy tqdm
3.  python build_rag_prompts.py
"""

import json
import os
import re
import zipfile
from collections import Counter
from io import TextIOWrapper
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm import tqdm

# ───────────────────────── CONFIGURATION ────────────────────────── #
CVE_ZIP = "cve_data.zip"          # zipped CVE list (optional)
CVE_DIR = "cve_data"              # extracted CVE list (alt. location)
OUTPUT_JSONL = "rag_zephyr_inputs.jsonl"

YEARS = set(range(2010, 2024))    # 2010 – 2023 inclusive
MAX_VOCAB = 5_000                 # TF‑IDF vocabulary size
NUM_CONTEXT = 5                   # k nearest neighbours per query
ENCODING = "utf‑8"

SEVERITY_MAP = {
    "LOW": "LOW",
    "MODERATE": "MEDIUM",         # CNA sometimes uses “MODERATE”
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "CRITICAL",
}

TOK_RE = re.compile(r"\w+")


def tokenize(txt: str):
    return TOK_RE.findall(txt.lower())


def safe_json_load(fp):
    """Robust JSON loader that survives BOMs or bad encodings."""
    try:
        return json.load(fp)
    except UnicodeDecodeError:
        fp.seek(0)
        return json.loads(fp.read().decode(ENCODING, "replace"))


# ───────────────────────── PART 1 – LOAD DATA ───────────────────── #
records = []

# Decide data source (ZIP vs folder) -------------------------------- #
if Path(CVE_ZIP).is_file():
    archive = zipfile.ZipFile(CVE_ZIP)
    members = [m for m in archive.namelist() if m.endswith(".json")]
    in_zip, closer = True, archive.close
    opener = lambda m: TextIOWrapper(archive.open(m), ENCODING)
elif Path(CVE_DIR).is_dir():
    members = [str(p) for p in Path(CVE_DIR).rglob("*.json")]
    in_zip, closer = False, lambda: None
    opener = lambda p: open(p, "r", encoding=ENCODING, errors="replace")
else:
    raise FileNotFoundError(
        "No data found. Place a zipped 'cve_data.zip' or extract the folder "
        "to './cve_data/'."
    )

for member in tqdm(members, desc="Extracting CVEs"):
    # Identify year from path …/cves/<year>/…
    parts = PurePosixPath(member).parts if in_zip else Path(member).parts
    try:
        year = int(parts[parts.index("cves") + 1])
    except (ValueError, IndexError):
        continue
    if year not in YEARS:
        continue

    # Load JSON
    try:
        with opener(member) as fp:
            data = safe_json_load(fp)
    except Exception:
        continue

    # ID
    cve_id = (
        data.get("cveMetadata", {}).get("cveId")
        or data.get("cveMetadata", {}).get("cveID")
    )
    if not cve_id:
        continue

    # English description
    desc = ""
    for d in data.get("containers", {}).get("cna", {}).get("descriptions", []):
        if d.get("lang", "en") == "en":
            desc = d.get("value", "").strip()
            break
    if not desc:
        continue

    # Severity (cvssV3_1 or cvssV3)
    sev = None
    for k in ("cvssV3_1", "cvssV3"):
        if sev:
            break
        for m in data.get("containers", {}).get("cna", {}).get("metrics", []):
            if k in m:
                sev = m[k].get("baseSeverity") or m[k].get("baseScore")
                break
    sev = SEVERITY_MAP.get(str(sev).upper())
    if sev is None:
        continue

    records.append({"CVE_ID": cve_id, "Description": desc, "Severity": sev})

closer()
if not records:
    raise RuntimeError("No usable CVE records found!")

# ─────────────────── PART 2 – BUILD TF‑IDF MATRIX ────────────────── #
token_lists = [tokenize(r["Description"]) for r in records]

df = Counter()
for tokens in token_lists:
    df.update(set(tokens))

vocab = [t for t, _ in df.most_common(MAX_VOCAB)]
v2i = {t: i for i, t in enumerate(vocab)}
N, V = len(token_lists), len(vocab)

idf = {t: np.log(N / (1 + df[t])) for t in vocab}

M = np.zeros((N, V), dtype=np.float32)
for i, tokens in enumerate(token_lists):
    tf = Counter(tokens)
    L = len(tokens) or 1
    for t, cnt in tf.items():
        j = v2i.get(t)
        if j is not None:
            M[i, j] = (cnt / L) * idf[t]

# Row‑wise L2 normalisation (version‑agnostic)
row_norms = np.linalg.norm(M, axis=1, keepdims=True)
row_norms[row_norms == 0] = 1.0
M /= row_norms

# ────────────── PART 3 – RETRIEVAL & PROMPT BUILDING ─────────────── #
def retrieve(desc: str, k: int = NUM_CONTEXT):
    tf = Counter(tokenize(desc))
    L = len(tf) or 1
    q = np.zeros((V,), dtype=np.float32)
    for t, cnt in tf.items():
        j = v2i.get(t)
        if j is not None:
            q[j] = (cnt / L) * idf[t]
    q /= np.linalg.norm(q) or 1.0
    idx = np.argsort(-(M @ q))[:k]
    return [records[i] for i in idx]


def build_prompt(query_desc: str, ctx_records):
    ctx = "\n".join(
        f"{e['CVE_ID']}: {e['Description']} (Severity: {e['Severity']})"
        for e in ctx_records
    )
    return (
        "You are a cybersecurity vulnerability classification expert.\n\n"
        "Reference examples:\n"
        f"{ctx}\n\n"
        "-----\n"
        "Classify the following CVE description:\n"
        f"{query_desc}\n\n"
        "Respond with exactly one of: LOW, MEDIUM, HIGH, CRITICAL.\n"
        "Severity:"
    )


# ────────────────────── PART 4 – WRITE OUTPUT ────────────────────── #
with open(OUTPUT_JSONL, "w", encoding=ENCODING) as out:
    for rec in tqdm(records, desc="Building RAG prompts"):
        prompt = build_prompt(rec["Description"], retrieve(rec["Description"]))
        out.write(json.dumps({"CVE_ID": rec["CVE_ID"], "prompt": prompt}) + "\n")

print(f"Saved {len(records):,} prompts to {OUTPUT_JSONL}")
