import io

import numpy as np
import pandas as pd
import pytest

from patchwork import demo, leakage, power, stats
from patchwork.audit import AuditConfig, parse_fitzpatrick, run_audit
from patchwork.web import create_app


# ------------------------------------------------------------------ stats
def test_auroc_matches_rank_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 1)  # many ties
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert stats.auroc(y, s) == pytest.approx(brute)


def test_auroc_edge_cases():
    assert stats.auroc([1, 0], [0.9, 0.1]) == 1.0
    assert stats.auroc([1, 0], [0.1, 0.9]) == 0.0
    assert np.isnan(stats.auroc([1, 1], [0.2, 0.3]))


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = y * 0.8 + rng.normal(0, 1, 400)
    point, lo, hi, samples = stats.bootstrap_auroc(y, s, n_boot=500)
    assert lo < point < hi
    assert len(samples) == 500


def test_ece_perfect_and_bad():
    assert stats.expected_calibration_error([0, 1], [0.0, 1.0]) == pytest.approx(0)
    assert stats.expected_calibration_error([0, 0], [1.0, 1.0]) == pytest.approx(1)


def test_majority_floor():
    assert stats.majority_floor([0, 0, 0, 1]) == 0.75


# ------------------------------------------------------------------ audit
@pytest.mark.parametrize("raw,expected", [
    (1, 1), ("6", 6), ("III", 3), ("Fitzpatrick IV", 4), ("FST5", 5), ("vi", 6), ("", None), (None, None), (7, None),
])
def test_parse_fitzpatrick(raw, expected):
    assert parse_fitzpatrick(raw) == expected


def test_demo_audit_detects_disparity():
    r = run_audit(demo.predictions())
    assert r["verdict"]["level"] == "bad"
    light, dark = r["contrast"]
    assert light["auroc"] > dark["auroc"]
    assert dark["includes_chance"]
    assert r["gap"]["ci_low"] > 0


def test_equal_groups_not_called_disparity():
    rng = np.random.default_rng(5)
    n = 4000
    y = (rng.random(n) < 0.4).astype(int)
    s = 1 / (1 + np.exp(-(1.2 * y + rng.normal(0, 1, n))))
    tone = rng.integers(1, 7, n)
    r = run_audit(pd.DataFrame({"label": y, "score": s, "fitzpatrick": tone}), AuditConfig(n_boot=400))
    assert r["verdict"]["level"] in {"good", "warn"}


def test_small_dark_group_is_inconclusive_not_fair():
    rng = np.random.default_rng(6)
    rows = []
    for band, n in ((2, 800), (5, 40)):
        y = (rng.random(n) < 0.3).astype(int)
        s = y * 1.0 + rng.normal(0, 1, n)
        rows.append(pd.DataFrame({"label": y, "score": s, "fitzpatrick": band}))
    r = run_audit(pd.concat(rows), AuditConfig(n_boot=400))
    assert r["verdict"]["level"] == "warn"
    assert "not evidence of fairness" in r["verdict"]["text"]


def test_missing_column_message():
    with pytest.raises(ValueError, match="Missing column"):
        run_audit(pd.DataFrame({"y": [0, 1] * 10, "score": [0.1] * 20, "fitzpatrick": [1] * 20}))


def test_logits_are_accepted():
    df = demo.predictions()
    df["score"] = np.log(df["score"] / (1 - df["score"]))
    r = run_audit(df, AuditConfig(n_boot=200))
    assert r["overall"]["auroc"] == run_audit(demo.predictions(), AuditConfig(n_boot=200))["overall"]["auroc"]


# ---------------------------------------------------------------- leakage
def test_leakage_flags_descriptor_and_patient_overlap():
    s = leakage.scan(demo.metadata_dev(), "label", "patient_id", "split")
    flagged = {c["column"] for c in s["columns"] if c["level"] == "bad"}
    assert "descriptor_plaque" in flagged
    assert "sex" not in flagged
    assert s["overlap"]["in_multiple_splits"] > 0


def test_shift_check_finds_head_neck_flip():
    sh = leakage.shift_check(demo.metadata_dev(), demo.metadata_ext(), "label", "body_site")
    head = next(r for r in sh["rows"] if r["category"] == "head_neck")
    assert head["lean_dev"] < 0 < head["lean_ext"]
    assert not head["agrees"]


# ------------------------------------------------------------------ power
def test_power_grows_with_sample_size():
    small = power.gap_power(0.71, 0.55, (40, 400), (18, 90))
    big = power.gap_power(0.71, 0.55, (400, 4000), (180, 900))
    assert small < big
    assert big > 0.8


def test_ci_width_matches_paper_scale():
    # 18 positive / 88 negative darker-skin cases gave a ~0.30-wide interval in the paper
    assert power.ci_width(0.503, 18, 88) == pytest.approx(0.30, abs=0.04)


def test_cases_needed_reaches_target():
    need = power.cases_needed(0.71, 0.60, 0.084, 0.17, 0.19)
    assert need["power"] >= 0.8


# -------------------------------------------------------------------- web
@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("url", ["/", "/audit", "/audit/demo", "/leakage", "/leakage/demo", "/power", "/about",
                                 "/samples/predictions.csv", "/healthz", "/power?auc_light=0.8&auc_dark=0.8"])
def test_pages_render(client, url):
    assert client.get(url).status_code == 200


def test_upload_audit(client):
    csv = demo.predictions().rename(columns={"fitzpatrick": "skin_type"}).to_csv(index=False).encode()
    resp = client.post("/audit", data={"file": (io.BytesIO(csv), "p.csv"), "label_col": "label",
                                        "score_col": "score", "group_col": "skin_type", "split_at": "4"},
                       content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Disparity detected" in resp.data


def test_upload_bad_file(client):
    resp = client.post("/audit", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert b"Missing column" in resp.data


def test_upload_leakage_with_shift(client):
    dev = demo.metadata_dev().to_csv(index=False).encode()
    ext = demo.metadata_ext().to_csv(index=False).encode()
    resp = client.post("/leakage", data={"file": (io.BytesIO(dev), "d.csv"), "external": (io.BytesIO(ext), "e.csv"),
                                          "label_col": "label", "patient_col": "patient_id", "split_col": "split",
                                          "shift_col": "body_site"}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"flips" in resp.data
