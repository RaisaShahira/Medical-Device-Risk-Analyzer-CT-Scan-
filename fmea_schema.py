from __future__ import annotations
import pandas as pd

FMEA_COLUMNS = [
    "Item/Function",
    "Failure Mode",
    "Effects of Failure",
    "Potential Cause(s)",
    "Severity (S)",
    "Occurrence (O)",
    "Detection (D)",
    "RPN",
    "Recommended Actions",
]

def calc_rpn(s: int, o: int, d: int) -> int:
    s = int(s)
    o = int(o)
    d = int(d)
    return s * o * d


def empty_fmea_df() -> pd.DataFrame:
    """
    Create empty dataframe with correct columns
    """
    return pd.DataFrame(columns=FMEA_COLUMNS)


def normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure S / O / D numeric.
    If user doesn't fill → default 1.
    Clip 1–10.
    Recalculate RPN always.
    """
    for col in ["Severity (S)", "Occurrence (O)", "Detection (D)"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(1)
                .astype(int)
                .clip(1, 10)
            )

    # Always recompute RPN
    df["RPN"] = [
        calc_rpn(s, o, d)
        for s, o, d in zip(
            df["Severity (S)"],
            df["Occurrence (O)"],
            df["Detection (D)"]
        )
    ]
    return df