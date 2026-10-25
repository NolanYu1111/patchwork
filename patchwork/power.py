"""How many cases does a fairness evaluation need?

Uses the Hanley & McNeil (1982) standard error for AUROC, which needs only the
expected AUROC and the number of positive and negative cases. Treating the two
skin-tone groups as independent, the standard error of the gap is the root sum
of squares, and power is the normal-approximation probability that the 95%
interval for the gap excludes zero.
"""

from __future__ import annotations

import math

from scipy.stats import norm

Z = 1.959964


def auroc_se(auc: float, n_pos: int, n_neg: int) -> float:
    if n_pos < 1 or n_neg < 1:
        return float("inf")
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(var, 1e-12))


def ci_width(auc: float, n_pos: int, n_neg: int) -> float:
    return 2 * Z * auroc_se(auc, n_pos, n_neg)


def gap_power(auc_light: float, auc_dark: float, light: tuple[int, int], dark: tuple[int, int]) -> float:
    """Probability that a 95% interval for the gap excludes zero, if the gap is real."""
    se = math.hypot(auroc_se(auc_light, *light), auroc_se(auc_dark, *dark))
    if not math.isfinite(se):
        return 0.0
    d = abs(auc_light - auc_dark) / se
    return float(norm.cdf(d - Z) + norm.cdf(-d - Z))


def cases_needed(auc_light, auc_dark, light_prev, dark_prev, ratio_dark, target=0.8, max_total=200_000):
    """Smallest total sample (split by ``ratio_dark``) reaching ``target`` power."""
    if abs(auc_light - auc_dark) < 1e-6:
        return None

    def power_at(total):
        n_dark = max(int(round(total * ratio_dark)), 2)
        n_light = max(total - n_dark, 2)
        lp = max(int(round(n_light * light_prev)), 1)
        dp = max(int(round(n_dark * dark_prev)), 1)
        return gap_power(auc_light, auc_dark, (lp, n_light - lp), (dp, n_dark - dp)), n_light, n_dark, lp, dp

    lo, hi = 10, 20
    while power_at(hi)[0] < target:
        hi *= 2
        if hi > max_total:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power_at(mid)[0] >= target:
            hi = mid
        else:
            lo = mid
    p, n_light, n_dark, lp, dp = power_at(hi)
    return {"total": hi, "n_light": n_light, "n_dark": n_dark, "pos_light": lp, "pos_dark": dp, "power": round(p, 3)}


def plan(auc_light, auc_dark, n_light, n_dark, light_prev, dark_prev) -> dict:
    lp = max(int(round(n_light * light_prev)), 0)
    dp = max(int(round(n_dark * dark_prev)), 0)
    light = (lp, n_light - lp)
    dark = (dp, n_dark - dp)
    power = gap_power(auc_light, auc_dark, light, dark)
    total = n_light + n_dark
    need = cases_needed(auc_light, auc_dark, light_prev, dark_prev, n_dark / total if total else 0.5)
    balanced = cases_needed(auc_light, auc_dark, light_prev, dark_prev, 0.5)
    curve = []
    for mult in (0.25, 0.5, 1, 2, 4, 8):
        nl, nd = max(int(n_light * mult), 2), max(int(n_dark * mult), 2)
        a, b = max(int(round(nl * light_prev)), 1), max(int(round(nd * dark_prev)), 1)
        curve.append({"scale": mult, "n_dark": nd, "power": round(gap_power(auc_light, auc_dark, (a, nl - a), (b, nd - b)), 3)})
    return {
        "inputs": {
            "auc_light": auc_light, "auc_dark": auc_dark, "n_light": n_light, "n_dark": n_dark,
            "light_prev": light_prev, "dark_prev": dark_prev,
        },
        "positives": {"light": lp, "dark": dp},
        "width_light": round(ci_width(auc_light, *light), 3),
        "width_dark": round(ci_width(auc_dark, *dark), 3),
        "power": round(power, 3),
        "needed_same_mix": need,
        "needed_balanced": balanced,
        "curve": curve,
    }
