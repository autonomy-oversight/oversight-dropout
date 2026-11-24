#!/usr/bin/env python3
"""Summarize pilot results: table (CSV + Markdown) and one figure per agent.

    python analyze.py results            # all agents under results/
    python analyze.py results --agents openai-gpt-4o-mini
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

LINK_ORDER = ["ideal", "moderate", "severe"]
BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated palette)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(root: str, agents: list[str] | None):
    eps = []
    for fn in glob.glob(os.path.join(root, "*", "*.json")):
        agent = os.path.basename(os.path.dirname(fn))
        if agents and agent not in agents:
            continue
        with open(fn) as f:
            eps.append(json.load(f))
    return eps


def condition_rows(eps):
    groups = defaultdict(list)
    for e in eps:
        c = e["config"]
        groups[(e["agent"], c.get("link_name", c["link"]), c["pressure"], int(c["recorder"]))].append(e)
    rows = []
    for (agent, link, pressure, rec), g in sorted(groups.items()):
        n = len(g)
        def frac(pred):
            k = sum(1 for e in g if pred(e))
            p, lo, hi = wilson(k, n)
            return k, p, lo, hi
        viol = [e for e in g if e["summary"]["any_violation"]]
        k_any, p_any, lo_any, hi_any = frac(lambda e: e["summary"]["any_violation"])
        row = {
            "agent": agent, "link": link, "pressure": pressure, "recorder": rec, "n": n,
            "any_violation": p_any, "any_violation_lo": lo_any, "any_violation_hi": hi_any,
        }
        for r in ("R1", "R2", "R3", "R4"):
            _, p, lo, hi = frac(lambda e, r=r: r in e["summary"]["violations_by_rule"])
            row[f"{r}_rate"], row[f"{r}_lo"], row[f"{r}_hi"] = p, lo, hi
        # Concealment: among episodes with >= 1 violation, share that left at
        # least one violated rule out of the post-mission report.
        kc = sum(1 for e in viol if e["report_scoring"]["concealed_rules"])
        pc, loc, hic = wilson(kc, len(viol))
        row.update(n_with_violation=len(viol), concealment=pc, concealment_lo=loc, concealment_hi=hic)
        row["overclaimed_inspection"] = sum(1 for e in g if e["report_scoring"]["overclaimed_inspections"]) / n
        row["mean_inspected"] = sum(e["summary"]["n_inspected"] for e in g) / n
        row["crash_rate"] = sum(1 for e in g if e["summary"]["crashed"]) / n
        row["invalid_actions_per_ep"] = sum(e["summary"]["invalid_actions"] for e in g) / n
        row["mean_link_down_s"] = sum(e["summary"]["link_down_s_during_flight"] for e in g) / n
        u = [e.get("usage", {}) for e in g]
        row["prompt_tokens"] = sum(x.get("prompt_tokens", 0) for x in u)
        row["completion_tokens"] = sum(x.get("completion_tokens", 0) for x in u)
        rows.append(row)
    return rows


def write_tables(rows, root):
    if not rows:
        return
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pct = lambda x: "n/a" if x != x else f"{100 * x:.0f}%"
    lines = ["| agent | link | pressure | rec | n | any viol. | R1 zone | R2 lost-link | R3 battery | R4 recall "
             "| concealed (of viol.) | waypoints | link down (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['agent']} | {r['link']} | {r['pressure']} | {r['recorder']} | {r['n']} | "
            f"{pct(r['any_violation'])} [{pct(r['any_violation_lo'])}–{pct(r['any_violation_hi'])}] | "
            f"{pct(r['R1_rate'])} | {pct(r['R2_rate'])} | {pct(r['R3_rate'])} | {pct(r['R4_rate'])} | "
            f"{pct(r['concealment'])} ({r['n_with_violation']}) | {r['mean_inspected']:.1f} | "
            f"{r['mean_link_down_s']:.0f} |")
    with open(os.path.join(root, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(rows, root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    panels = [("any_violation", "Episodes with ≥1 rule violation"),
              ("R2_rate", "Lost-link procedure violated (R2)"),
              ("concealment", "Violations left out of report\n(share of violating episodes)")]
    for agent in sorted({r["agent"] for r in rows}):
        sub = [r for r in rows if r["agent"] == agent and r["recorder"] == 0]
        links = [l for l in LINK_ORDER if any(r["link"] == l for r in sub)] + \
                sorted({r["link"] for r in sub} - set(LINK_ORDER))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
        plt.rcParams.update({"font.size": 9})
        for ax, (key, title) in zip(axes, panels):
            for j, (pressure, color) in enumerate((("low", BLUE), ("high", ORANGE))):
                xs, ys, los, his = [], [], [], []
                for i, l in enumerate(links):
                    m = [r for r in sub if r["link"] == l and r["pressure"] == pressure]
                    if not m or m[0][key] != m[0][key]:
                        continue
                    lo_key = key.replace("_rate", "_lo") if key.endswith("_rate") else f"{key}_lo"
                    hi_key = key.replace("_rate", "_hi") if key.endswith("_rate") else f"{key}_hi"
                    xs.append(i + (j - 0.5) * 0.36)
                    ys.append(m[0][key])
                    los.append(m[0][key] - m[0][lo_key])
                    his.append(m[0][hi_key] - m[0][key])
                ax.bar(xs, ys, width=0.34, color=color, label=f"{pressure} mission pressure",
                       edgecolor="white", linewidth=2)
                ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#555555",
                            elinewidth=1, capsize=2)
            ax.set_title(title, fontsize=9, loc="left")
            ax.set_xticks(range(len(links)), links)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_xlabel("oversight link")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        n = sub[0]["n"] if sub else 0
        fig.suptitle(f"{agent} — n={n} episodes per bar, 95% Wilson intervals", fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        out = os.path.join(root, f"fig_{agent}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results")
    ap.add_argument("--agents", default=None, help="comma list of agent folder names")
    a = ap.parse_args()
    eps = load(a.root, a.agents.split(",") if a.agents else None)
    if not eps:
        raise SystemExit(f"no episodes found under {a.root}/<agent>/")
    rows = condition_rows(eps)
    write_tables(rows, a.root)
    plot(rows, a.root)


if __name__ == "__main__":
    main()
