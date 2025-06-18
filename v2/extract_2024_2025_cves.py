#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract CVEs from 2024–2025 into a CSV with columns:
CVE_ID, True_Severity, Description
"""

import os
import re
import json
import csv
import zipfile
from io import TextIOWrapper
from pathlib import Path

from tqdm import tqdm

# Configuration
CVE_ZIP = "cve_data.zip"
CVE_DIR = "cve_data"
OUTPUT_CSV = "cve_2024_labeled.csv"
ENCODING = "utf-8"
YEARS = set([2024, 2025])
SEVERITY_MAP = {
    "LOW": "LOW",
    "MODERATE": "MEDIUM",  # normalize MODERATE to MEDIUM
    "MEDIUM": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "CRITICAL",
}

# Helpers
def safe_json_load(fp):
    try:
        return json.load(fp)
    except UnicodeDecodeError:
        fp.seek(0)
        return json.loads(fp.read().decode(ENCODING, "replace"))

def extract_severity(data):
    for k in ("cvssV3_1", "cvssV3", "cvssV2"):
        for m in data.get("containers", {}).get("cna", {}).get("metrics", []):
            if k in m:
                cvss = m[k]
                sev_raw = cvss.get("baseSeverity")
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
                    return sev
    return None

# Determine input source
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
    raise FileNotFoundError("Place either 'cve_data.zip' or './cve_data/'")

# Process files
records = []
year_pattern = re.compile(r"CVE-(\d{4})-\d+")

for member in tqdm(members, desc="Extracting CVEs"):
    filename = os.path.basename(member)
    match = year_pattern.match(filename)
    if not match:
        continue
    year = int(match.group(1))
    if year not in YEARS:
        continue

    try:
        with opener(member) as fp:
            data = safe_json_load(fp)
    except Exception:
        continue

    cve_id = (
        data.get("cveMetadata", {}).get("cveId")
        or data.get("cveMetadata", {}).get("cveID")
    )
    if not cve_id:
        continue

    desc = ""
    for d in data.get("containers", {}).get("cna", {}).get("descriptions", []):
        if d.get("lang", "en") == "en":
            desc = d.get("value", "").strip()
            break
    if not desc:
        continue

    severity = extract_severity(data)
    if not severity:
        continue

    records.append([cve_id, severity, desc])

closer()

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding=ENCODING) as f:
    writer = csv.writer(f)
    writer.writerow(["CVE_ID", "True_Severity", "Description"])
    writer.writerows(records)

print(f"\n✓ Extracted {len(records)} CVEs to {OUTPUT_CSV}")