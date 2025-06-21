#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Graph-RAG CVE Classifier 🚀
- GPU-accelerated embeddings
- Batched query processing
- Parallel inference
- Cached graph + embeddings
"""

import argparse, csv, json, os, re, time, pickle, zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from statistics import mode
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor

# ─────── CONFIG ───────
CVE_ZIP = "cve_data.zip"
CVE_DIR = "cve_data"
YEARS = set(range(2010, 2024))
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"
LORA_DIR = "zephyr_lora"
EDGE_THR = 0.60
K_PER_NODE = 8
MAX_HOPS = 2
MAX_CTX = 3
N_PASSES = 1
TEMP = 0.25
MAX_TOK = 6
QUERY_CSV = "cve_2024_labeled.csv"
SAVE_DIR = "graph_rag_real"
os.makedirs(SAVE_DIR, exist_ok=True)

SEV_MAP = {"LOW": "LOW", "MODERATE": "MEDIUM", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}
CRIT_KEYS = ("remote code execution", "unauthenticated", "root access", "arbitrary command", "buffer overflow", "heap overflow", "out-of-bounds write", "memory corruption", "privilege escalation", "sql injection")
LOW_KEYS = ("information disclosure only", "path disclosure", "username enumeration", "cross-site scripting", "open redirect", "clickjacking")

# ─────── Graph Construction ───────
def iter_cve_json():
    if Path(CVE_ZIP).is_file():
        with zipfile.ZipFile(CVE_ZIP) as z:
            for n in z.namelist():
                if n.endswith(".json") and any(str(y) in n for y in YEARS):
                    yield z.read(n).decode("utf-8", "replace")
    elif Path(CVE_DIR).is_dir():
        for p in Path(CVE_DIR).rglob("*.json"):
            if any(str(y) in str(p) for y in YEARS):
                yield p.read_text("utf-8", "replace")
    else:
        raise FileNotFoundError("No CVE data found")

def parse_one(raw):
    try:
        j = json.loads(raw)
        cid = j.get("cveMetadata", {}).get("cveId")
        cna = j.get("containers", {}).get("cna", {})
        desc = next((d.get("value", "") for d in cna.get("descriptions", []) if d.get("lang") == "en"), "").strip()
        for k in ("cvssV3_1", "cvssV3"):
            for m in cna.get("metrics", []):
                if k in m:
                    sev = m[k].get("baseSeverity") or m[k].get("baseScore")
                    break
        return cid, desc, SEV_MAP.get(str(sev).upper()) if sev else None
    except:
        return None

def build_graph(emb_model):
    ids, descs, sevs = [], [], []
    for raw in tqdm(iter_cve_json(), desc="Reading CVEs"):
        row = parse_one(raw)
        if row and all(row):
            ids.append(row[0])
            descs.append(row[1])
            sevs.append(row[2])

    embeds = emb_model.encode(descs, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    embeds_tensor = torch.tensor(np.array(embeds))
    G = nx.Graph()
    G.add_nodes_from(range(len(ids)))

    for i in tqdm(range(len(ids)), desc="Adding edges"):
        sims = util.cos_sim(embeds_tensor[i].unsqueeze(0), embeds_tensor)[0]
        topk = torch.topk(sims, K_PER_NODE + 1)
        for sim_val, j in zip(topk.values.tolist(), topk.indices.tolist()):
            if i != j and sim_val >= EDGE_THR:
                G.add_edge(i, j, weight=float(sim_val))

    np.save(f"{SAVE_DIR}/embeds.npy", embeds_tensor.numpy())
    with open(f"{SAVE_DIR}/graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    return ids, descs, sevs, G, embeds_tensor

def neighborhood(seeds, G):
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(MAX_HOPS):
        nxt = set()
        for n in frontier:
            nxt.update(G.neighbors(n))
        seen.update(nxt)
        frontier = nxt
    return list(seen - set(seeds))

def build_prompt(desc, nbh_ids, ids, descs, sevs):
    high = [j for j in nbh_ids if sevs[j] == "HIGH"]
    crit = [j for j in nbh_ids if sevs[j] == "CRITICAL"]
    others = [j for j in nbh_ids if j not in high + crit]
    chosen = (others[:MAX_CTX - 2] + high[:1] + crit[:1])[:MAX_CTX]
    ctx = "\n".join(f"{ids[j]}: {descs[j]} (Severity: {sevs[j]})" for j in chosen)
    cue = ("• remote code execution  • unauthenticated attacker • root access\n"
           "• arbitrary command execution  • buffer/heap overflow  • OOB write\n")
    return ("You are a cybersecurity vulnerability classification expert.\n\n"
            f"Reference examples:\n{ctx}\n"
            f"Common phrases that imply HIGH/CRITICAL:\n{cue}"
            "-----\nClassify the following CVE description:\n"
            f"{desc}\nRespond with exactly one of: LOW, MEDIUM, HIGH, CRITICAL.\nSeverity:")

def parse_label(txt):
    m = re.search(r"\b(LOW|MEDIUM|HIGH|CRITICAL)\b", txt.upper())
    return m.group(1) if m else "UNKNOWN"

def apply_heur(desc, pred):
    d = desc.lower()
    if any(k in d for k in CRIT_KEYS) and pred in {"MEDIUM", "HIGH"}:
        return "CRITICAL"
    if any(k in d for k in LOW_KEYS) and pred in {"HIGH", "CRITICAL"}:
        return "MEDIUM"
    return pred

def generate_label(prompt, tok, model):
    t = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**t, max_new_tokens=MAX_TOK, temperature=TEMP)
    return parse_label(tok.decode(out[0], skip_special_tokens=True))

def main(limit):
    t0 = time.time()
    emb_model = SentenceTransformer(EMB_MODEL, device='cuda')

    if Path(f"{SAVE_DIR}/graph.gpickle").exists():
        with open(f"{SAVE_DIR}/graph.gpickle", "rb") as f:
            G = pickle.load(f)
        embeds_tensor = torch.tensor(np.load(f"{SAVE_DIR}/embeds.npy"))
        with open(f"{SAVE_DIR}/ids_descs_sevs.pkl", "rb") as f:
            ids, descs, sevs = pickle.load(f)
    else:
        ids, descs, sevs, G, embeds_tensor = build_graph(emb_model)
        with open(f"{SAVE_DIR}/ids_descs_sevs.pkl", "wb") as f:
            pickle.dump((ids, descs, sevs), f)

    dfq = pd.read_csv(QUERY_CSV).dropna(subset=["Description", "True_Severity"]).reset_index(drop=True)
    if limit:
        dfq = dfq.head(limit)

    query_descs = dfq["Description"].tolist()
    q_embeds = emb_model.encode(query_descs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    sim_matrix = util.cos_sim(torch.tensor(q_embeds), embeds_tensor)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, LORA_DIR).eval()

    results = []
    for i, row in enumerate(tqdm(dfq.to_dict("records"), desc="Classifying")):
        sims = sim_matrix[i]
        topk = torch.topk(sims, K_PER_NODE).indices.tolist()
        nbh = neighborhood(topk, G)
        nbh_ids = (topk + nbh)[: 3 * K_PER_NODE]
        prompt = build_prompt(row["Description"], nbh_ids, ids, descs, sevs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(generate_label, prompt, tok, model) for _ in range(N_PASSES)]
            labels = [f.result() for f in futures]

        if labels:
            pred = mode([apply_heur(row["Description"], label) for label in labels])
            results.append(dict(CVE_ID=row["CVE_ID"], True_Severity=row["True_Severity"].upper(), Predicted_Severity=pred))

    df = pd.DataFrame(results)
    df.to_csv(f"{SAVE_DIR}/results.csv", index=False)
    acc = accuracy_score(df.True_Severity, df.Predicted_Severity)
    rep = classification_report(df.True_Severity, df.Predicted_Severity, labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"], zero_division=0)
    with open(f"{SAVE_DIR}/eval.txt", "w") as fp:
        fp.write(rep + f"\nOverall accuracy: {acc:.2%}")
    print(rep)
    print(f"DONE — accuracy {acc:.2%}, total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N CVEs for quick test (0 = all)")
    main(ap.parse_args().limit)
