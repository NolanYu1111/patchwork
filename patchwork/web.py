"""Flask web interface for Patchwork."""

from __future__ import annotations

import io
import json

import pandas as pd
from flask import Flask, Response, render_template, request

from . import __version__, charts, demo, leakage, power
from .audit import AuditConfig, run_audit

MAX_UPLOAD_MB = 10


def _read_csv(storage) -> pd.DataFrame:
    if storage is None or not storage.filename:
        raise ValueError("Choose a CSV file to upload.")
    raw = storage.read()
    if not raw:
        raise ValueError("The uploaded file is empty.")
    try:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read that file as CSV ({exc}).") from exc


def _float(name, default, lo, hi):
    try:
        v = float(request.args.get(name, default))
    except ValueError:
        v = default
    return min(max(v, lo), hi)


def _int(name, default, lo, hi):
    try:
        v = int(float(request.args.get(name, default)))
    except ValueError:
        v = default
    return min(max(v, lo), hi)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    @app.context_processor
    def _globals():
        return {"version": __version__, "path": request.path}

    @app.errorhandler(413)
    def too_big(_):
        return render_template("audit_form.html", error=f"File is larger than {MAX_UPLOAD_MB} MB."), 413

    @app.get("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------ audit
    @app.get("/audit")
    def audit_form():
        return render_template("audit_form.html")

    def _render_audit(report, source):
        return render_template(
            "audit_result.html",
            r=report,
            source=source,
            roc_svg=charts.roc(report["curves"], report["contrast"]),
            forest_svg=charts.forest(report["bands"], report["contrast"], report["config"]["split_at"]),
            report_json=json.dumps({k: v for k, v in report.items() if k != "curves"}, indent=2),
        )

    @app.post("/audit")
    def audit_run():
        try:
            df = _read_csv(request.files.get("file"))
            cfg = AuditConfig(
                label_col=request.form.get("label_col", "label").strip(),
                score_col=request.form.get("score_col", "score").strip(),
                group_col=request.form.get("group_col", "fitzpatrick").strip(),
                split_at=int(request.form.get("split_at", 4)),
            )
            report = run_audit(df, cfg)
        except ValueError as exc:
            return render_template("audit_form.html", error=str(exc)), 400
        return _render_audit(report, request.files["file"].filename)

    @app.get("/audit/demo")
    def audit_demo():
        return _render_audit(run_audit(demo.predictions()), "Simulated demo cohort (no real patients)")

    # ---------------------------------------------------------------- leakage
    @app.get("/leakage")
    def leakage_form():
        return render_template("leakage_form.html")

    def _render_leakage(scan, shift, source):
        return render_template("leakage_result.html", s=scan, shift=shift, source=source)

    @app.post("/leakage")
    def leakage_run():
        try:
            df = _read_csv(request.files.get("file"))
            label = request.form.get("label_col", "label").strip()
            pid = request.form.get("patient_col", "").strip() or None
            split = request.form.get("split_col", "").strip() or None
            scan = leakage.scan(df, label, pid, split)
            shift = None
            ext = request.files.get("external")
            shift_col = request.form.get("shift_col", "").strip()
            if ext is not None and ext.filename and shift_col:
                shift = leakage.shift_check(df, _read_csv(ext), label, shift_col)
        except ValueError as exc:
            return render_template("leakage_form.html", error=str(exc)), 400
        return _render_leakage(scan, shift, request.files["file"].filename)

    @app.get("/leakage/demo")
    def leakage_demo():
        dev, ext = demo.metadata_dev(), demo.metadata_ext()
        return _render_leakage(
            leakage.scan(dev, "label", "patient_id", "split"),
            leakage.shift_check(dev, ext, "label", "body_site"),
            "Simulated development and external cohorts (no real patients)",
        )

    # ------------------------------------------------------------------ power
    @app.get("/power")
    def power_page():
        p = power.plan(
            auc_light=_float("auc_light", 0.71, 0.5, 0.99),
            auc_dark=_float("auc_dark", 0.60, 0.5, 0.99),
            n_light=_int("n_light", 465, 10, 1_000_000),
            n_dark=_int("n_dark", 106, 10, 1_000_000),
            light_prev=_float("light_prev", 0.084, 0.01, 0.99),
            dark_prev=_float("dark_prev", 0.17, 0.01, 0.99),
        )
        return render_template("power.html", p=p, curve_svg=charts.power_curve(p["curve"], p["inputs"]["n_dark"]))

    # ------------------------------------------------------------------ misc
    @app.get("/about")
    def about():
        return render_template("about.html")

    @app.get("/samples/<name>.csv")
    def sample(name):
        makers = {
            "predictions": demo.predictions,
            "metadata_development": demo.metadata_dev,
            "metadata_external": demo.metadata_ext,
        }
        if name not in makers:
            return Response("Not found", 404)
        csv = makers[name]().to_csv(index=False)
        return Response(csv, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=patchwork_{name}.csv"})

    @app.get("/healthz")
    def health():
        return {"ok": True, "version": __version__}

    return app
