"""Reproduce every number, table and figure of the paper.

    python scripts/run_all.py              # synthetic, physically based flight sample
    python scripts/run_all.py --measured   # use data/measured/flights.csv instead

Outputs: paper/generated/*.tex (numbers.tex macros and table bodies) and paper/figures/*.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bandmodel import flights, params, report, system  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "parameters.yaml"))
    ap.add_argument("--measured", action="store_true", help="use observed flights instead of the generator")
    args = ap.parse_args()

    P = params.load(args.config)
    df_syn, cal = flights.generate(P)
    if args.measured:
        df = flights.load_measured(P)
        label = "observed"
    else:
        df = df_syn
        label = "synthetic"
    base = system.build_base(P, df)
    out = report.build_all(P, df, cal, base, label)

    summary = {
        "data": label,
        "calibration_max_rel_residual": cal.residual,
        "index_2050": {k: float(r.index[-1]) for k, r in out["scenarios"].items()},
        "shapley_2050_Mt": out["shapley_2050"].sum().round(2).to_dict(),
        "corrective_cost_bn": {f"{k[0]} {k[1]}": round(v["cost_eur"] / 1e9, 2) for k, v in out["corrective"].items()},
    }
    (ROOT / "paper" / "generated" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
