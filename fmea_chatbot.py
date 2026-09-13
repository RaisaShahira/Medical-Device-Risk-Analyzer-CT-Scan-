from __future__ import annotations
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm import LLMClient
from similarity_engine import FailureModeSimilarityEngine


CHAT_SYSTEM = """You are a safety/risk assistant for a CT Scan Medical-Device-Risk-Analyzer.
Answer ONLY using the provided FMEA table context. If the answer isn't in the table, say:
"Not found in the saved FMEA table."
Keep answers concise and practical.
"""


def df_to_documents(df: pd.DataFrame) -> List[str]:
    docs = []
    for i, row in df.iterrows():
        docs.append(
            f"Row {i+1} | Item/Function: {row.get('Item/Function','')}\n"
            f"Failure Mode: {row.get('Failure Mode','')}\n"
            f"Effects: {row.get('Effects of Failure','')}\n"
            f"Cause: {row.get('Potential Cause(s)','')}\n"
            f"S/O/D/RPN: {row.get('Severity (S)','')}/{row.get('Occurrence (O)','')}/{row.get('Detection (D)','')} RPN={row.get('RPN','')}\n"
            f"Rec Actions: {row.get('Recommended Actions','')}\n"
        )
    return docs


def retrieve_topk(df: pd.DataFrame, question: str, k: int = 4) -> List[str]:
    docs = df_to_documents(df)
    if len(docs) == 0:
        return []

    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(docs)

    q = vect.transform([question])
    sims = cosine_similarity(q, X).flatten()
    idx = np.argsort(-sims)[:min(k, len(docs))]

    return [docs[i] for i in idx]


def answer_question(llm: LLMClient, df: pd.DataFrame, question: str) -> str:
    
    # 1️⃣ normal retrieval dulu
    ctx_docs = retrieve_topk(df, question, k=4)

    # 2️⃣ cek similarity dataset
    dataset_match = engine.find_best_match(question, top_k=1)

    if dataset_match:
        m = dataset_match[0]
        return f"""
Ditemukan kasus mirip di database FMEA:

Failure Mode: {m.get('Failure Mode', '')}
Effect: {m.get('Effects of Failure', '')}
Cause: {m.get('Potential Cause(s)', '')}
Recommended Action: {m.get('Recommended Actions', '')}
(Similarity Score: {m.get('similarity_score', 0):.2f})
"""

    # 3️⃣ kalau tidak ada match → pakai LLM
    ctx = "\n\n---\n\n".join(ctx_docs) if ctx_docs else "(empty)"

    user = f"""Saved FMEA context:
{ctx}

User question: {question}
"""
    return llm.generate_text(system=CHAT_SYSTEM, user=user)

engine = FailureModeSimilarityEngine("FMEACT.csv")
