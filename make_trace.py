#!/usr/bin/env python3
"""Turn an aerial link measurement log into a 1 Hz up/down trace for replay,
and print the Gilbert-Elliott parameters that match it.

The input is any CSV with a time column (seconds, or ISO timestamps) and a
link-quality column (e.g. SINR, RSRP, RSRQ, throughput, or a 0/1 'connected'
flag). A second counts as "up" if the metric meets the threshold in at least
one sample in that second (use --rule mean to average instead).

    python make_trace.py flight1.csv --time-col time --metric-col sinr_db \
        --threshold -3 --out traces/flight1.csv
    python run.py --agent llm --model ... --links ideal,trace:traces/flight1.csv

Seconds without any sample count as "down" (no measurement = no link) unless
--gap up is given.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime

from oversight_dropout.link import outage_stats


def parse_time(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--metric-col", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--below", action="store_true", help="link is up when metric <= threshold (e.g. latency)")
    ap.add_argument("--rule", choices=["any", "mean"], default="any")
    ap.add_argument("--gap", choices=["down", "up"], default="down")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    per_sec: dict[int, list[float]] = {}
    with open(a.input, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t, m = parse_time(row[a.time_col]), float(row[a.metric_col])
            except (ValueError, KeyError, TypeError):
                continue
            if math.isnan(m):
                continue
            per_sec.setdefault(int(math.floor(t)), []).append(m)
    if not per_sec:
        raise SystemExit("no usable rows")
    ok = (lambda m: m <= a.threshold) if a.below else (lambda m: m >= a.threshold)
    t0, t1 = min(per_sec), max(per_sec)
    ups = []
    for s in range(t0, t1 + 1):
        vals = per_sec.get(s)
        if not vals:
            ups.append(a.gap == "up")
        elif a.rule == "mean":
            ups.append(ok(sum(vals) / len(vals)))
        else:
            ups.append(any(ok(v) for v in vals))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "up"])
        for i, u in enumerate(ups):
            w.writerow([i, int(u)])

    st = outage_stats(ups)
    good_runs, cur = [], 0
    for u in ups:
        if u:
            cur += 1
        elif cur:
            good_runs.append(cur)
            cur = 0
    if cur:
        good_runs.append(cur)
    mean_good = sum(good_runs) / len(good_runs) if good_runs else float("inf")
    print(f"wrote {a.out}: {len(ups)} s, {100 * st['frac_down']:.1f}% down, "
          f"{st['n_outages']} outages, mean {st['mean_outage_s']:.1f} s, max {st['max_outage_s']} s")
    print(f"matching Gilbert-Elliott preset: GEParams(mean_good_s={mean_good:.1f}, "
          f"mean_bad_s={st['mean_outage_s']:.1f})")


if __name__ == "__main__":
    main()
