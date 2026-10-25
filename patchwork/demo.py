"""Synthetic demo data.

These tables are *simulated*. They are generated to reproduce the shape of the
external-validation results in the paper (lighter-skin AUROC about 0.71, darker-skin
AUROC near chance, a leaky clinician-descriptor column, and a body-site association
that flips between cohorts) so the app can be tried without any real patient data.
No patient images or records are included anywhere in Patchwork.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .audit import ROMAN_OF


def _scores(rng, y, auc):
    # binormal model: positives shifted by d so that AUROC = Phi(d / sqrt 2)
    d = np.sqrt(2) * norm.ppf(auc)
    z = rng.normal(0, 1, len(y)) + d * y
    return 1 / (1 + np.exp(-(z - d / 2 - 1.6)))


def predictions(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = [  # band, n, positives, simulated AUROC
        (1, 53, 4, 0.70), (2, 170, 13, 0.68), (3, 150, 12, 0.69), (4, 92, 10, 0.74),
        (5, 62, 11, 0.52), (6, 44, 7, 0.48),
    ]
    rows = []
    for band, n, pos, auc in plan:
        y = np.r_[np.ones(pos, int), np.zeros(n - pos, int)]
        s = _scores(rng, y, auc)
        rows += [{"case_id": f"SIM-{band}-{i:03d}", "label": int(a), "score": round(float(b), 4),
                  "fitzpatrick": ROMAN_OF[band]} for i, (a, b) in enumerate(zip(y, s))]
    # a few cases without a skin-type label, as in real cohorts
    for i in range(12):
        y = int(rng.random() < 0.1)
        rows.append({"case_id": f"SIM-U-{i:03d}", "label": y, "score": round(float(_scores(rng, np.array([y]), 0.6)[0]), 4),
                     "fitzpatrick": ""})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


SITES = ["head_neck", "trunk", "arms", "hands", "legs", "feet", "buttocks", "genital", "scalp", "nails"]


def _metadata(rng, n, pos_rate, site_lean, with_descriptors):
    y = (rng.random(n) < pos_rate).astype(int)
    rows = []
    for i, yi in enumerate(y):
        weights = np.array([max(0.02, 0.1 + (site_lean[s] if yi else -site_lean[s])) for s in SITES])
        site = rng.choice(SITES, p=weights / weights.sum())
        row = {
            "patient_id": f"P{i // 2:04d}",
            "label": int(yi),
            "age": int(np.clip(rng.normal(34 + 6 * yi, 15), 1, 90)),
            "sex": rng.choice(["female", "male"]),
            "body_site": site,
        }
        if with_descriptors:
            row["descriptor_plaque"] = int(rng.random() < (0.85 if yi else 0.12))
            row["descriptor_lichenification"] = int(rng.random() < (0.10 if yi else 0.55))
        rows.append(row)
    return pd.DataFrame(rows)


def metadata_dev(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lean = {s: 0.0 for s in SITES}
    lean.update(head_neck=-0.08, scalp=-0.06, trunk=0.05, legs=0.06, buttocks=0.04, hands=-0.03)
    df = _metadata(rng, 900, 0.36, lean, with_descriptors=True)
    df["split"] = np.where(rng.random(len(df)) < 0.8, "train", "test")  # image-level split: leaks patients
    return df


def metadata_ext(seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lean = {s: 0.0 for s in SITES}
    lean.update(head_neck=0.12, scalp=0.08, trunk=0.05, legs=0.05, buttocks=0.04, hands=0.03)
    return _metadata(rng, 700, 0.10, lean, with_descriptors=False)
