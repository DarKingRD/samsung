import json
import re
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# =====================
# PATHS
# =====================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

KARTASLOV_DIR = RAW_DIR / "kartaslov"
GITHUB_DIR = RAW_DIR / "github"
LORUGEC_DIR = RAW_DIR / "loru"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================
# FILTERS
# =====================

CODE_TOKENS = [
    "{", "}", ";", "::", "=>", "<", ">", "\\", "%", "$",
    "function", "class", "import", "return", "const", "var",
    "def ", "=", "==", "->", "</", "/>"
]

def is_russian_text(text: str) -> bool:
    cyr = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
    latin = sum(c.isascii() and c.isalpha() for c in text)
    return cyr >= 3 and cyr > latin

def looks_like_code(text: str) -> bool:
    text = text.lower()
    return any(tok in text for tok in CODE_TOKENS)

def is_good_text(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    if not is_russian_text(text):
        return False
    if looks_like_code(text):
        return False
    return True

# =====================
# KARTASLOV
# =====================

def process_kartaslov():
    rows = []

    for csv_path in KARTASLOV_DIR.glob("*.csv"):
        df = pd.read_csv(csv_path, sep=";")

        for _, row in df.iterrows():
            correct = str(row["CORRECT"]).strip()
            mistake = str(row["MISTAKE"]).strip()
            weight = float(row.get("WEIGHT", 1.0))

            if not is_good_text(correct):
                continue

            rows.append({
                "task": "spell",
                "input": mistake,
                "target": correct,
                "source": "kartaslov",
                "weight": weight,
            })

    out = PROCESSED_DIR / "kartaslov_spell.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] Kartaslov → {len(rows)} rows")

# =====================
# GITHUB
# =====================

def process_github():
    rows = []
    jsonl_path = GITHUB_DIR / "github-typo-corpus.v1.0.0.jsonl"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="GitHub"):
            data = json.loads(line)

            for edit in data.get("edits", []):
                src = edit["src"]["text"].strip()
                tgt = edit["tgt"]["text"].strip()

                if src == tgt:
                    continue

                if not is_good_text(src):
                    continue

                rows.append({
                    "task": "spell",
                    "input": src,
                    "target": tgt,
                    "source": "github",
                    "weight": 1.0,
                })

    out = PROCESSED_DIR / "github_spell.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] GitHub → {len(rows)} rows")

# =====================
# LORUGEC
# =====================

def process_lorugec():
    rows = []
    xlsx = LORUGEC_DIR / "LORuGEC.xlsx"
    df = pd.read_excel(xlsx)

    for _, row in df.iterrows():
        src = str(row["Initial sentence"]).strip()
        tgt = str(row["Correct sentence"]).strip()

        if src == tgt:
            continue

        if not is_good_text(src):
            continue

        rows.append({
            "task": "grammar",
            "input": src,
            "target": tgt,
            "source": "lorugec",
            "weight": 1.0,
        })

    out = PROCESSED_DIR / "lorugec_grammar.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] LORuGEC → {len(rows)} rows")

# =====================
# MERGE
# =====================

def merge_all():
    dfs = []

    for file in PROCESSED_DIR.glob("*.csv"):
        if file.name == "all_train.csv":
            continue
        dfs.append(pd.read_csv(file))

    all_df = pd.concat(dfs).sample(frac=1.0).reset_index(drop=True)

    out = PROCESSED_DIR / "all_train.csv"
    all_df.to_csv(out, index=False)

    print(f"[OK] MERGED → {len(all_df)} rows")

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    process_kartaslov()
    process_github()
    process_lorugec()
    merge_all()
