"""Metadata provenance and leakage checks.

Two checks motivated by the paper:

1. **Leakage scan.** Clinician-recorded morphologic descriptors lifted metadata
   AUROC from 0.795 to 0.894 because the same clinician who assigned the label
   recorded them. A feature that predicts the label *too* well, or whose name looks
   like a diagnostic descriptor, is flagged for provenance review. Patient-level
   overlap between splits is also checked.

2. **Shift check.** Body-site associations reversed between cohorts (head and neck
   leaned eczema by 23 points internally and psoriasis by 8 points externally).
   Given a second cohort, each category's association with the label is compared
   and inversions are flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import auroc

DESCRIPTOR_HINTS = (
    "plaque", "scale", "scaling", "lichen", "morpholog", "descriptor", "diagnos", "lesion_type",
    "biopsy", "histo", "treatment", "medication", "drug", "steroid", "ddx", "differential",
    "erythema", "papule", "vesic", "annotat", "label", "class", "icd",
)
ID_HINTS = ("patient", "subject", "person", "case_id", "image_id", "file", "path", "id")

HIGH_AUROC = 0.85


def _oof_target_encode(x: pd.Series, y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold category means, so a category's own label never scores itself."""
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, k, size=len(x))
    prior = y.mean()
    enc = np.full(len(x), prior, dtype=float)
    x = x.astype(str).to_numpy()
    for f in range(k):
        tr, te = folds != f, folds == f
        means = pd.Series(y[tr]).groupby(x[tr]).agg(["sum", "count"])
        # light smoothing toward the prior
        smoothed = (means["sum"] + 5 * prior) / (means["count"] + 5)
        enc[te] = pd.Series(x[te]).map(smoothed).fillna(prior).to_numpy()
    return enc


def column_signal(col: pd.Series, y: np.ndarray) -> float | None:
    """Out-of-fold univariate AUROC for one column, folded so 0.5 is no signal."""
    mask = col.notna().to_numpy()
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return None
    c, yy = col[mask], y[mask]
    numeric = pd.to_numeric(c, errors="coerce")
    if numeric.notna().mean() > 0.95 and numeric.nunique() > 10:
        a = auroc(yy[numeric.notna().to_numpy()], numeric.dropna().to_numpy())
    else:
        a = auroc(yy, _oof_target_encode(c, yy))
    if a is None or np.isnan(a):
        return None
    return float(max(a, 1 - a))


def scan(df: pd.DataFrame, label_col: str, patient_col: str | None = None, split_col: str | None = None) -> dict:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {', '.join(map(str, df.columns))}")
    y_raw = pd.to_numeric(df[label_col], errors="coerce")
    keep = y_raw.isin([0, 1])
    if keep.sum() < 20:
        raise ValueError("Need at least 20 rows with a 0/1 label.")
    df = df[keep].reset_index(drop=True)
    y = y_raw[keep].astype(int).to_numpy()

    skip = {label_col, patient_col, split_col}
    rows = []
    for name in df.columns:
        if name in skip:
            continue
        lname = str(name).lower()
        s = column_signal(df[name], y)
        flags = []
        if any(h in lname for h in DESCRIPTOR_HINTS):
            flags.append("Name suggests a clinician-recorded or diagnostic field")
        if s is not None and s >= HIGH_AUROC:
            flags.append(f"Predicts the label unusually well on its own (AUROC {s:.2f})")
        nunique = df[name].nunique(dropna=True)
        if nunique > 0.9 * len(df) and any(h in lname for h in ID_HINTS):
            flags.append("Looks like an identifier; identifiers should never be model inputs")
        named_descriptor = any(h in lname for h in DESCRIPTOR_HINTS)
        if (s is not None and s >= HIGH_AUROC) or (named_descriptor and s is not None and s > 0.7):
            level = "bad"
        elif flags:
            level = "warn"
        else:
            level = "ok"
        rows.append({
            "column": str(name),
            "auroc": None if s is None else round(s, 3),
            "missing": round(float(df[name].isna().mean()), 3),
            "distinct": int(nunique),
            "flags": flags,
            "level": level,
        })
    rows.sort(key=lambda r: -(r["auroc"] or 0))

    overlap = None
    if patient_col and split_col and patient_col in df.columns and split_col in df.columns:
        per_patient = df.groupby(patient_col)[split_col].nunique()
        leaked = per_patient[per_patient > 1]
        overlap = {
            "patients": int(per_patient.shape[0]),
            "in_multiple_splits": int(leaked.shape[0]),
            "examples": [str(p) for p in leaked.index[:10]],
        }

    dupes = int(df.drop(columns=[c for c in (patient_col,) if c and c in df.columns]).duplicated().sum())
    return {
        "n_rows": int(len(df)),
        "positive_rate": round(float(y.mean()), 3),
        "columns": rows,
        "overlap": overlap,
        "duplicate_rows": dupes,
    }


def shift_check(a: pd.DataFrame, b: pd.DataFrame, label_col: str, column: str, min_count: int = 10) -> dict:
    """Compare each category's lean toward the positive class across two cohorts."""
    for name, d in (("development", a), ("external", b)):
        for c in (label_col, column):
            if c not in d.columns:
                raise ValueError(f"Column '{c}' missing from the {name} cohort.")

    def lean(d):
        y = pd.to_numeric(d[label_col], errors="coerce")
        d = d[y.isin([0, 1])]
        y = y[y.isin([0, 1])].astype(int)
        base = y.mean()
        g = pd.DataFrame({"cat": d[column].astype(str).str.strip().str.lower(), "y": y}).groupby("cat")["y"]
        stats = g.agg(["mean", "count"])
        stats["lean"] = stats["mean"] - base
        return stats, base

    sa, base_a = lean(a)
    sb, base_b = lean(b)
    shared = sorted(set(sa.index) & set(sb.index))
    rows = []
    for cat in shared:
        na, nb = int(sa.loc[cat, "count"]), int(sb.loc[cat, "count"])
        la, lb = float(sa.loc[cat, "lean"]), float(sb.loc[cat, "lean"])
        enough = na >= min_count and nb >= min_count
        # Leans within +/- 2 points are treated as neutral rather than a direction.
        neutral = abs(la) < 0.02 or abs(lb) < 0.02
        agree = neutral or np.sign(la) == np.sign(lb)
        rows.append({
            "category": cat,
            "n_dev": na,
            "n_ext": nb,
            "lean_dev": round(la * 100, 1),
            "lean_ext": round(lb * 100, 1),
            "agrees": bool(agree),
            "neutral": bool(neutral),
            "enough": enough,
        })
    rows.sort(key=lambda r: (r["agrees"], -abs(r["lean_dev"] - r["lean_ext"])))
    judged = [r for r in rows if r["enough"]]
    return {
        "column": column,
        "base_dev": round(base_a, 3),
        "base_ext": round(base_b, 3),
        "rows": rows,
        "only_dev": sorted(set(sa.index) - set(sb.index)),
        "only_ext": sorted(set(sb.index) - set(sa.index)),
        "agree_count": sum(r["agrees"] for r in judged),
        "judged_count": len(judged),
    }
