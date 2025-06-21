#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REAL Graph-RAG CVE Classifier 🌐 (LoRA-Zephyr) – **FIXED**

• Embedding-graph neighbourhood retrieval (no exact-string matching)
• Works for any query description, including unseen 2024 CVEs
• Outputs results.csv + eval.txt in graph_rag_real/
"""

import argparse, csv, json, os, re, zipfile, warnings, time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mode
import networkx as nx

import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pickle

warnings.filterwarnings("ignore", category=UserWarning)

# ───────── CONFIG ─────────
CVE_ZIP    = "cve_data.zip"
CVE_DIR    = "cve_data"
YEARS      = set(range(2010, 2024))
EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"
BASE_MODEL = "HuggingFaceH4/zephyr-7b-beta"
LORA_DIR   = "zephyr_lora"

EDGE_THR   = 0.60      # cosine threshold for an edge
K_PER_NODE = 8         # edges per node in the graph
MAX_HOPS   = 2         # depth for neighbour expansion

MAX_CTX    = 3         # examples in each prompt
N_PASSES   = 1         # self-consistency votes
TEMP       = 0.25
MAX_TOK    = 6         # model response length

QUERY_CSV  = "cve_2024_labeled.csv"
SAVE_DIR   = "graph_rag_real"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Heuristic keywords (unchanged) ───
CRIT_KEYS = ("remote code execution","unauthenticated","root access",
             "arbitrary command","buffer overflow","heap overflow",
             "out-of-bounds write","memory corruption",
             "privilege escalation","sql injection")
LOW_KEYS  = ("information disclosure only","path disclosure",
             "username enumeration","cross-site scripting",
             "open redirect","clickjacking")

SEV_MAP = {"LOW":"LOW","MODERATE":"MEDIUM","MEDIUM":"MEDIUM",
           "HIGH":"HIGH","CRITICAL":"CRITICAL"}

# ───────── Data Loading ─────────
def iter_cve_json():
    if Path(CVE_ZIP).is_file():
        with zipfile.ZipFile(CVE_ZIP) as z:
            for n in z.namelist():
                if not n.endswith(".json"):
                    continue
                parts = PurePosixPath(n).parts
                try:
                    yr = int(parts[parts.index("cves") + 1])
                except Exception:
                    continue
                if yr in YEARS:
                    yield z.read(n).decode("utf-8", "replace")
    elif Path(CVE_DIR).is_dir():
        for p in Path(CVE_DIR).rglob("*.json"):
            parts = p.parts
            try:
                yr = int(parts[parts.index("cves") + 1])
            except Exception:
                continue
            if yr in YEARS:
                yield p.read_text("utf-8", "replace")
    else:
        raise FileNotFoundError("No CVE data found.")

def parse_one(raw):
    try:
        j = json.loads(raw)
    except Exception:
        return None
    cna = j.get("containers", {}).get("cna", {})
    cid = j.get("cveMetadata", {}).get("cveId")
    if not cid:
        return None
    # description
    desc = ""
    for d in cna.get("descriptions", []):
        if d.get("lang", "en") == "en":
            desc = re.sub(r"\s+", " ", d.get("value", "")).strip()
            break
    if not desc:
        return None
    # severity
    sev = None
    for k in ("cvssV3_1", "cvssV3"):
        for m in cna.get("metrics", []):
            if k in m:
                sev = m[k].get("baseSeverity") or m[k].get("baseScore")
                break
    sev = SEV_MAP.get(str(sev).upper()) if sev else None
    if not sev:
        return None
    return cid, desc, sev

# ───────── Build Graph ─────────

def build_graph():
    ids, descs, sevs = [], [], []
    for raw in tqdm(iter_cve_json(), desc="Reading CVEs"):
        row = parse_one(raw)
        if row:
            ids.append(row[0])
            descs.append(row[1])
            sevs.append(row[2])

    emb_model = SentenceTransformer(EMB_MODEL)
    embeds = emb_model.encode(
        descs,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeds_tensor = torch.from_numpy(np.array(embeds))

    G = nx.Graph()
    G.add_nodes_from(range(len(ids)))

    for i in tqdm(range(len(ids)), desc="Adding edges"):
        sims = util.cos_sim(
            embeds_tensor[i].unsqueeze(0), embeds_tensor
        )[0]
        topk = torch.topk(sims, K_PER_NODE + 1)  # +1 to include self
        for sim_val, j in zip(topk.values.tolist(), topk.indices.tolist()):
            if i == j or sim_val < EDGE_THR:
                continue
            G.add_edge(i, j, weight=float(sim_val))

    np.save(f"{SAVE_DIR}/embeds.npy", embeds_tensor.numpy())
    with open(f"{SAVE_DIR}/graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    return ids, descs, sevs, G, embeds_tensor

# ───────── Prompt utils ─────────
def neighborhood(seed_nodes, G):
    """Return 1- & 2-hop neighbours for a *set* of seed nodes."""
    seen = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(MAX_HOPS):
        nxt = set()
        for n in frontier:
            nxt.update(G.neighbors(n))
        seen.update(nxt)
        frontier = nxt
    return list(seen - set(seed_nodes))

def build_prompt(desc, nbh_ids, ids, descs, sevs):
    # choose up to MAX_CTX refs with at least 1 HIGH + 1 CRIT when possible
    high = [j for j in nbh_ids if sevs[j] == "HIGH"]
    crit = [j for j in nbh_ids if sevs[j] == "CRITICAL"]
    others = [j for j in nbh_ids if j not in high + crit]
    chosen = (others[: MAX_CTX - 2] + high[:1] + crit[:1])[:MAX_CTX]

    ctx = "\n".join(
        f"{ids[j]}: {descs[j]} (Severity: {sevs[j]})" for j in chosen
    )
    cue = (
        "• remote code execution  • unauthenticated attacker • root access\n"
        "• arbitrary command execution  • buffer/heap overflow  • OOB write\n"
    )
    return (
        "You are a cybersecurity vulnerability classification expert.\n\n"
        f"Reference examples:\n{ctx}\n"
        f"Common phrases that imply HIGH/CRITICAL:\n{cue}"
        "-----\nClassify the following CVE description:\n"
        f"{desc}\n"
        "Respond with exactly one of: LOW, MEDIUM, HIGH, CRITICAL.\nSeverity:"
    )

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

# ───────── Main ─────────
def main(limit):
    t0 = time.time()
    ids, descs, sevs, G, embeds_tensor = build_graph()
    graph_build_secs = time.time() - t0

    dfq = (
        pd.read_csv(
            QUERY_CSV,
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
            encoding_errors="replace",
        )
        .dropna(subset=["Description", "True_Severity"])
        .reset_index(drop=True)
    )
    if limit:
        dfq = dfq.head(limit)

    # embedding model for queries (cheap re-load, few secs)
    emb_model_q = SentenceTransformer(EMB_MODEL)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(base, LORA_DIR).eval()

    results = []
    for row in tqdm(dfq.to_dict("records"), desc="Classifying"):
        # find k nearest graph nodes to *this query* by embedding similarity
        q_emb = emb_model_q.encode(
        row["Description"], normalize_embeddings=True
        )
        q_tensor = torch.tensor(q_emb).unsqueeze(0)
        sims = util.cos_sim(q_tensor, embeds_tensor)[0]
        topk = torch.topk(sims, K_PER_NODE).indices.tolist()

        nbh = neighborhood(topk, G)
        nbh_ids = (topk + nbh)[: 3 * K_PER_NODE]  # cap prompt size

        votes = []
        for _ in range(N_PASSES):
            prompt = build_prompt(
                row["Description"], nbh_ids, ids, descs, sevs
            )
            t = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **t, max_new_tokens=MAX_TOK, temperature=TEMP
            )
            label = parse_label(
                tok.decode(out[0], skip_special_tokens=True)
            )
            votes.append(apply_heur(row["Description"], label))

        if votes:  # shouldn’t ever be empty
            pred = mode(votes)
            results.append(
                dict(
                    CVE_ID=row["CVE_ID"],
                    True_Severity=row["True_Severity"].upper(),
                    Predicted_Severity=pred,
                )
            )

    if not results:
        print("⚠️  No predictions produced – check input data.")
        return

    df = pd.DataFrame(results)
    df.to_csv(f"{SAVE_DIR}/results.csv", index=False)

    acc = accuracy_score(df.True_Severity, df.Predicted_Severity)
    rep = classification_report(
        df.True_Severity,
        df.Predicted_Severity,
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        zero_division=0,
    )
    with open(f"{SAVE_DIR}/eval.txt", "w") as fp:
        fp.write(rep + f"\nOverall accuracy: {acc:.2%}")

    print(rep)
    print(
        f"DONE — accuracy {acc:.2%} "
        f"(graph build {graph_build_secs:.1f}s, total {time.time()-t0:.1f}s)"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="first N CVEs for quick test (0 = all)",
    )
    main(ap.parse_args().limit)

