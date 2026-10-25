"""Skin-tone-stratified audit of a binary classifier's predictions.

Input is a table with one row per test case: the true label (0/1), the model's
score for the positive class, and a Fitzpatrick skin type. Output is a plain
dictionary that the web app, the CLI and the JSON export all share.

The audit follows the protocol in Yu, Yu & Homayouni (IEEE BigData 2026 HSS):
AUROC rather than accuracy, bootstrap intervals rather than significance tests,
Fitzpatrick I–IV vs V–VI as the primary contrast, and plain-language findings
that never treat "no detected gap" as "fair".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import stats

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
ROMAN_OF = {v: k for k, v in ROMAN.items()}

# Below these counts an interval is too wide to say much; the thresholds mirror
# the sample sizes discussed in the paper (18 darker-skin positives gave a
# 0.30-wide interval).
MIN_POSITIVES = 30
MIN_CASES = 100


@dataclass
class AuditConfig:
    label_col: str = "label"
    score_col: str = "score"
    group_col: str = "fitzpatrick"
    split_at: int = 4  # bands <= split_at are "lighter", > split_at are "darker"
    n_boot: int = 2000
    seed: int = 0


def parse_fitzpatrick(value) -> int | None:
    """Turn 3, '3', 'III', 'Fitzpatrick IV', 'FST5' into an integer 1–6, else None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if 1 <= int(value) <= 6 else None
    if isinstance(value, (float, np.floating)):
        return int(value) if float(value).is_integer() and 1 <= value <= 6 else None
    text = str(value).strip().upper()
    m = re.search(r"\b(VI|IV|V|I{1,3})\b", text)
    if m:
        return ROMAN[m.group(1)]
    m = re.search(r"([1-6])", text)
    return int(m.group(1)) if m else None


def _r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def _group_block(name, y, s, n_boot, seed):
    point, lo, hi, samples = stats.bootstrap_auroc(y, s, n_boot=n_boot, seed=seed)
    return {
        "group": name,
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "auroc": _r(point),
        "ci_low": _r(lo),
        "ci_high": _r(hi),
        "ci_width": _r(hi - lo) if not np.isnan(point) else None,
        "balanced_accuracy": _r(stats.balanced_accuracy(y, s)),
        "ece": _r(stats.expected_calibration_error(y, s)),
        "includes_chance": bool(not np.isnan(point) and lo <= 0.5 <= hi),
    }, samples


def load_table(df: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    missing = [c for c in (cfg.label_col, cfg.score_col, cfg.group_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s): {', '.join(missing)}. Columns found: {', '.join(map(str, df.columns))}"
        )
    out = pd.DataFrame(
        {
            "label": pd.to_numeric(df[cfg.label_col], errors="coerce"),
            "score": pd.to_numeric(df[cfg.score_col], errors="coerce"),
            "band": df[cfg.group_col].map(parse_fitzpatrick),
        }
    )
    out = out.dropna(subset=["label", "score"])
    labels = set(out["label"].unique())
    if not labels <= {0, 1}:
        raise ValueError(f"The label column must contain only 0 and 1; found {sorted(labels)}.")
    if out["score"].min() < 0 or out["score"].max() > 1:
        # Logits or arbitrary scores are fine for AUROC; squash for calibration only.
        out["score"] = 1 / (1 + np.exp(-out["score"]))
    out["label"] = out["label"].astype(int)
    return out


def run_audit(df: pd.DataFrame, cfg: AuditConfig | None = None) -> dict:
    cfg = cfg or AuditConfig()
    t = load_table(df, cfg)
    if len(t) < 10:
        raise ValueError("Need at least 10 rows with a label and a score.")
    if t["label"].nunique() < 2:
        raise ValueError("Both classes (0 and 1) must appear in the label column.")

    y_all, s_all = t["label"].to_numpy(), t["score"].to_numpy()
    overall, _ = _group_block("All cases", y_all, s_all, cfg.n_boot, cfg.seed)
    overall["majority_floor"] = _r(stats.majority_floor(y_all))

    known = t.dropna(subset=["band"])
    unknown_n = int(len(t) - len(known))

    # Per-band blocks (pattern only — usually too small to interpret alone)
    bands = []
    for b in range(1, 7):
        sub = known[known["band"] == b]
        if len(sub):
            block, _ = _group_block(ROMAN_OF[b], sub["label"].to_numpy(), sub["score"].to_numpy(), cfg.n_boot, cfg.seed + b)
            bands.append(block)
        else:
            bands.append({"group": ROMAN_OF[b], "n": 0, "positives": 0, "auroc": None})

    lighter_name = f"Fitzpatrick I–{ROMAN_OF[cfg.split_at]}"
    darker_name = f"Fitzpatrick {ROMAN_OF[cfg.split_at + 1]}–VI"
    light = known[known["band"] <= cfg.split_at]
    dark = known[known["band"] > cfg.split_at]
    lb, ls = _group_block(lighter_name, light["label"].to_numpy(), light["score"].to_numpy(), cfg.n_boot, cfg.seed + 11)
    db, ds = _group_block(darker_name, dark["label"].to_numpy(), dark["score"].to_numpy(), cfg.n_boot, cfg.seed + 12)

    gap = None
    if lb["auroc"] is not None and db["auroc"] is not None:
        lo, hi = stats.gap_interval(ls, ds, seed=cfg.seed + 13)
        gap = {"point": _r(lb["auroc"] - db["auroc"]), "ci_low": _r(lo), "ci_high": _r(hi)}

    fpr, tpr = stats.roc_curve_points(y_all, s_all)
    curves = {"All cases": (fpr.tolist(), tpr.tolist())}
    for name, sub in ((lighter_name, light), (darker_name, dark)):
        if sub["label"].nunique() == 2:
            f, tp = stats.roc_curve_points(sub["label"], sub["score"])
            curves[name] = (f.tolist(), tp.tolist())

    report = {
        "config": cfg.__dict__,
        "n_rows": int(len(t)),
        "unknown_tone": unknown_n,
        "overall": overall,
        "contrast": [lb, db],
        "gap": gap,
        "bands": bands,
        "curves": curves,
    }
    report["findings"] = findings(report)
    report["verdict"] = verdict(report)
    return report


def verdict(r: dict) -> dict:
    """One headline for the top of the report."""
    gap, (light, dark) = r["gap"], r["contrast"]
    if gap is None:
        return {
            "level": "unknown",
            "title": "Not enough skin-tone data to compare groups",
            "text": "Both skin-tone groups need cases from both classes before a gap can be estimated.",
        }
    if gap["ci_low"] > 0 or gap["ci_high"] < 0:
        worse = dark if gap["point"] > 0 else light
        return {
            "level": "bad",
            "title": f"Disparity detected: the model is weaker on {worse['group']}",
            "text": (
                f"The AUROC gap is {gap['point']:+.3f} (95% CI {gap['ci_low']:+.3f} to {gap['ci_high']:+.3f}). "
                "The interval excludes zero, so the data support a real difference in this direction."
            ),
        }
    under = any(g["positives"] < MIN_POSITIVES for g in (light, dark))
    if under or (gap["ci_high"] - gap["ci_low"]) > 0.15:
        return {
            "level": "warn",
            "title": "Inconclusive: a gap can neither be confirmed nor ruled out",
            "text": (
                f"The gap is {gap['point']:+.3f} but its interval runs from {gap['ci_low']:+.3f} to {gap['ci_high']:+.3f}. "
                "This is absence of evidence, not evidence of fairness. Collect more cases (see the Power planner)."
            ),
        }
    return {
        "level": "good",
        "title": "No disparity detected at this sample size",
        "text": (
            f"The gap is {gap['point']:+.3f} with a tight interval ({gap['ci_low']:+.3f} to {gap['ci_high']:+.3f}). "
            "Remember that one test set is one population: confirm on an independent external cohort."
        ),
    }


def findings(r: dict) -> list[dict]:
    out = []
    light, dark = r["contrast"]
    ov = r["overall"]

    for g in (light, dark):
        if g["n"] == 0:
            out.append({"level": "bad", "text": f"No cases in {g['group']}. The model cannot be evaluated on these skin tones."})
            continue
        if g["includes_chance"]:
            out.append({
                "level": "bad",
                "text": f"On {g['group']} the AUROC interval ({g['ci_low']}–{g['ci_high']}) includes 0.5: performance is indistinguishable from chance.",
            })
        if g["positives"] < MIN_POSITIVES:
            out.append({
                "level": "warn",
                "text": f"{g['group']} has only {g['positives']} positive cases. Intervals will be wide (current width {g['ci_width']}).",
            })

    missing = [b["group"] for b in r["bands"] if b["n"] == 0]
    if missing:
        out.append({
            "level": "warn",
            "text": f"No test cases for Fitzpatrick {', '.join(missing)}. Results say nothing about these skin types.",
        })

    if r["unknown_tone"]:
        share = r["unknown_tone"] / r["n_rows"]
        out.append({
            "level": "warn" if share > 0.2 else "info",
            "text": f"{r['unknown_tone']} cases ({share:.0%}) have no usable skin-type label and are excluded from the skin-tone comparison.",
        })

    if ov["ece"] is not None and ov["ece"] > 0.1:
        out.append({
            "level": "warn",
            "text": f"Expected calibration error is {ov['ece']}. Scores should not be read as probabilities or used with a fixed confidence cut-off.",
        })

    out.append({
        "level": "info",
        "text": f"Always guessing the majority class would already score {ov['majority_floor']:.0%} accuracy here, which is why Patchwork reports AUROC.",
    })
    out.append({
        "level": "info",
        "text": "This is one test population. In our study a disparity invisible on internal data appeared at once on an independent cohort, so repeat this audit externally.",
    })
    return out
