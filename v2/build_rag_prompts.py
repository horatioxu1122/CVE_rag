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
    "MODERATE": "MEDIUM",  # normalize MODERATE to MEDIUM
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "CRITICAL",
}

skipped_year_files = []

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

# Diagnostic counters
total_files = 0
skipped_by_year = 0
missing_cve_id = 0
missing_desc = 0
missing_severity = 0
parsed_successfully = 0

# Decide data source (ZIP vs folder) ------------------------------- #
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

year_pattern = re.compile(r"CVE-(\d{4})-\d+\.json", re.IGNORECASE)

for member in tqdm(members, desc="Extracting CVEs"):
    total_files += 1

    # Extract year from filename using regex
    filename = os.path.basename(member)
    match = year_pattern.match(filename)

    if not match:
        skipped_by_year += 1
        skipped_year_files.append(str(member))  # Add full path
        continue

    year = int(match.group(1))
    if year not in YEARS:
        skipped_by_year += 1
        skipped_year_files.append(str(member))
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
        missing_cve_id += 1
        continue

    # English description
    desc = ""
    for d in data.get("containers", {}).get("cna", {}).get("descriptions", []):
        if d.get("lang", "en") == "en":
            desc = d.get("value", "").strip()
            break
    if not desc:
        missing_desc += 1
        continue

    # Severity: supports cvssV3_1, cvssV3, and cvssV2; falls back to baseScore
    sev = None

    for k in ("cvssV3_1", "cvssV3", "cvssV2"):
        for m in data.get("containers", {}).get("cna", {}).get("metrics", []):
            if k in m:
                cvss = m[k]
                sev_raw = cvss.get("baseSeverity")

                # Fallback: derive severity from baseScore if baseSeverity is missing
                if not sev_raw and "baseScore" in cvss:
                    try:
                        score = float(cvss["baseScore"])
                        if score < 4.0:
                            sev_raw = "LOW"
                        elif score < 7.0:
                            sev_raw = "MEDIUM"
                        elif score < 9.0:
                            sev_raw = "HIGH"
                        else:
                            sev_raw = "CRITICAL"
                    except Exception:
                        continue

                sev = SEVERITY_MAP.get(str(sev_raw).upper())
                if sev:
                    break
        if sev:
            break

    # If still missing, skip and count
    if sev is None:
        missing_severity += 1
        # Optional: debug print
        if missing_severity <= 10:
            print(f"[DEBUG] Missing severity in: {cve_id}")
        continue

    records.append({"CVE_ID": cve_id, "Description": desc, "Severity": sev})
    parsed_successfully += 1

closer()
print("\n===== SUMMARY =====")
print(f"Total files scanned:           {total_files:,}")
print(f"Skipped (bad/missing year):    {skipped_by_year:,}")
print(f"Skipped (missing CVE ID):      {missing_cve_id:,}")
print(f"Skipped (missing description): {missing_desc:,}")
print(f"Skipped (missing severity):    {missing_severity:,}")
print(f"✓ Records retained:            {parsed_successfully:,}")
print("====================\n")

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

if skipped_year_files:
    with open("skipped_due_to_year.json", "w", encoding="utf-8") as f:
        json.dump(skipped_year_files, f, indent=2)
    print(f"Saved {len(skipped_year_files)} skipped year filenames to skipped_due_to_year.json")

with open(OUTPUT_JSONL, "w", encoding=ENCODING) as out:
    for rec in tqdm(records, desc="Building RAG prompts"):
        prompt = build_prompt(rec["Description"], retrieve(rec["Description"]))
        out.write(json.dumps({"CVE_ID": rec["CVE_ID"], "prompt": prompt}) + "\n")

print(f"Saved {len(records):,} prompts to {OUTPUT_JSONL}")
