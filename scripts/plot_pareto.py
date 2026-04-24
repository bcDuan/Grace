"""Plot k-budget vs metric from a CSV: columns k, metric, method."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV with k,metric,method")
    p.add_argument("--out", default="experiments/results/pareto.png")
    args = p.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required", file=sys.stderr)
        sys.exit(1)
    rows: list[dict[str, str]] = []
    with open(args.input, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print("empty csv", file=sys.stderr)
        sys.exit(1)
    by: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        m = str(r.get("method", "m"))
        k = float(r["k"])
        met = float(r["metric"])
        by.setdefault(m, []).append((k, met))
    for m, pts in by.items():
        pts.sort(key=lambda t: t[0])
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("k")
    plt.ylabel("metric")
    plt.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
