from __future__ import annotations
import json
import pandas as pd
from llm import LLMClient
from fmea_schema import FMEA_COLUMNS
from similarity_engine import FailureModeSimilarityEngine
from fmea_schema import normalize_scores

GEN_SYSTEM = """You are a biomedical risk engineer helping build an FMEA for a CT scanner.
Output must be valid JSON: a list of objects. Each object MUST contain:
- Item/Function
- Failure Mode
- Effects of Failure
- Potential Cause(s)
- Recommended Actions

Rules:
- Focus on CT scan workflow + patient safety + device malfunction.
- Do NOT fill Severity / Occurrence / Detection.
- If unsure, write "Unknown".
"""

def build_prompt(max_rows: int) -> str:
    return f"""
Generate up to {max_rows} realistic FMEA rows for a CT Scanner.
Make the scenarios relevant to real clinical use (patient positioning, radiation exposure,
image artifact, power failure, operator error, etc).
"""

def compute_rpn(row):
    try:
        s = int(row.get("Severity (S)", 0) or 0)
        o = int(row.get("Occurrence (O)", 0) or 0)
        d = int(row.get("Detection (D)", 0) or 0)

        if s and o and d:
            return s * o * d
        return 0
    except Exception:
        return 0


def generate_fmea_rows(llm: LLMClient, max_rows: int = 5) -> pd.DataFrame:
    user = build_prompt(max_rows=max_rows)
    raw = llm.generate_text(system=GEN_SYSTEM, user=user)

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("LLM output is not a list")
    except Exception:
        return pd.DataFrame(columns=FMEA_COLUMNS)

    rows = []

    for obj in data:
        row = {c: "" for c in FMEA_COLUMNS}

        # Fill AI generated
        for k in [
            "Item/Function",
            "Failure Mode",
            "Effects of Failure",
            "Potential Cause(s)",
            "Recommended Actions"
        ]:
            if k in obj:
                row[k] = obj[k]

        # User will fill later
        row["Severity (S)"] = ""
        row["Occurrence (O)"] = ""
        row["Detection (D)"] = ""

        # Auto RPN
        row["RPN"] = compute_rpn(row)

        rows.append(row)

    return pd.DataFrame(rows, columns=FMEA_COLUMNS)

# SIMILARITY ENGINE
sim_engine = FailureModeSimilarityEngine("FMEACT.csv")


def auto_fill_from_similarity(df: pd.DataFrame) -> pd.DataFrame:
    updated_rows = []

    for _, row in df.iterrows():
        fm = row.get("Failure Mode", "")

        match_list = sim_engine.find_best_match(fm)
        match = match_list[0] if match_list else None

        if match:
            row["Item/Function"] = match.get("Item/Function", row.get("Item/Function", ""))
            row["Effects of Failure"] = match.get("Effects of Failure", "")
            row["Potential Cause(s)"] = match.get("Potential Cause(s)", "")
            row["Recommended Actions"] = match.get("Recommended Actions", "")

            # default S O D biar ga kosong
            row["Severity (S)"] = row.get("Severity (S)", 1)
            row["Occurrence (O)"] = row.get("Occurrence (O)", 1)
            row["Detection (D)"] = row.get("Detection (D)", 1)

        updated_rows.append(row)

    df = pd.DataFrame(updated_rows, columns=df.columns)

    # normalize & recompute
    df = normalize_scores(df)
    return df
