"""Command-line use, for pipelines and CI.

    python -m patchwork audit predictions.csv --label label --score score --group fitzpatrick
    python -m patchwork leakage metadata.csv --label label --patient patient_id --split split
    python -m patchwork serve
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="patchwork", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="skin-tone-stratified AUROC audit of predictions")
    a.add_argument("csv")
    a.add_argument("--label", default="label")
    a.add_argument("--score", default="score")
    a.add_argument("--group", default="fitzpatrick")
    a.add_argument("--split-at", type=int, default=4)
    a.add_argument("--fail-on-disparity", action="store_true", help="exit 2 if a gap is detected (for CI)")

    lk = sub.add_parser("leakage", help="scan metadata columns for label leakage")
    lk.add_argument("csv")
    lk.add_argument("--label", default="label")
    lk.add_argument("--patient")
    lk.add_argument("--split")

    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--port", type=int, default=5000)

    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .web import create_app

        create_app().run(port=args.port)
        return 0

    df = pd.read_csv(args.csv, sep=None, engine="python")
    if args.cmd == "audit":
        from .audit import AuditConfig, run_audit

        r = run_audit(df, AuditConfig(args.label, args.score, args.group, args.split_at))
        r.pop("curves")
        json.dump(r, sys.stdout, indent=2, default=str)
        print()
        print(f"\n{r['verdict']['title']}\n{r['verdict']['text']}", file=sys.stderr)
        if args.fail_on_disparity and r["verdict"]["level"] == "bad":
            return 2
        return 0

    from .leakage import scan

    json.dump(scan(df, args.label, args.patient, args.split), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
