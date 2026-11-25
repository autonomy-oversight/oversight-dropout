#!/usr/bin/env python3
"""Summarize pilot results: table (CSV + Markdown), the pre-specified tests
of ANALYSIS_PLAN.md (stats.md), and one figure per agent.

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


# ------------------------------------------------------------------ statistics
# Pre-specified in ANALYSIS_PLAN.md. Episodes are paired by seed: for a given
# seed, every condition sees the same link timeline and the same recall draw.

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def upper_bound_zero(n: int, alpha: float = 0.05) -> float:
    """One-sided 95% upper bound on a rate when 0 events were seen in n trials."""
    return 1 - alpha ** (1 / n) if n else float("nan")


def paired_flip_test(diffs: list, n_perm: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test for a mean paired difference."""
    import random
    if not diffs or all(d == 0 for d in diffs):
        return 1.0
    obs = abs(sum(diffs))
    rng = random.Random(seed)
    hits = sum(1 for _ in range(n_perm)
               if abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12)
    return (hits + 1) / (n_perm + 1)


def _index(eps):
    idx = {}
    for e in eps:
        c = e["config"]
        idx[(c.get("link_name", c["link"]), c["pressure"], int(c["recorder"]), c["seed"])] = e
    return idx


def _paired(idx, a_key, b_key, outer):
    """Return list of (episode_a, episode_b) pairs. a_key/b_key map an outer
    tuple (value, seed) to an index key."""
    pairs = []
    for o in outer:
        ka, kb = a_key(*o), b_key(*o)
        if ka in idx and kb in idx:
            pairs.append((idx[ka], idx[kb]))
    return pairs


def _compare(pairs, label_a, label_b):
    viol = lambda e: bool(e["summary"]["any_violation"])
    b = sum(1 for x, y in pairs if viol(x) and not viol(y))
    c = sum(1 for x, y in pairs if viol(y) and not viol(x))
    n = len(pairs)
    ka = sum(viol(x) for x, _ in pairs)
    kb = sum(viol(y) for _, y in pairs)
    return {"n_pairs": n, "a": label_a, "b": label_b,
            "rate_a": ka / n if n else float("nan"), "rate_b": kb / n if n else float("nan"),
            "k_a": ka, "k_b": kb, "discordant_a_only": b, "discordant_b_only": c,
            "p": mcnemar_exact(b, c)}


def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def statistics(eps, root):
    lines = ["# Pre-specified tests (see ANALYSIS_PLAN.md)", ""]
    by_agent = defaultdict(list)
    for e in eps:
        by_agent[e["agent"]].append(e)
    for agent, g in sorted(by_agent.items()):
        idx = _index(g)
        seeds = sorted({k[3] for k in idx})
        prs = sorted({k[1] for k in idx})
        links = sorted({k[0] for k in idx})
        tests = {}
        tests["RQ1 link: severe vs ideal"] = _compare(_paired(
            idx, lambda p, s: ("severe", p, 0, s), lambda p, s: ("ideal", p, 0, s),
            [(p, s) for p in prs for s in seeds]), "severe", "ideal")
        tests["RQ2 pressure: high vs low"] = _compare(_paired(
            idx, lambda l, s: (l, "high", 0, s), lambda l, s: (l, "low", 0, s),
            [(l, s) for l in links for s in seeds]), "high", "low")
        tests["RQ3 recorder (severe link): off vs on"] = _compare(_paired(
            idx, lambda p, s: ("severe", p, 0, s), lambda p, s: ("severe", p, 1, s),
            [(p, s) for p in prs for s in seeds]), "recorder off", "recorder on")
        tests = {k: v for k, v in tests.items() if v["n_pairs"] > 0}
        adj = holm({k: v["p"] for k, v in tests.items()})

        lines += [f"## {agent}", "",
                  "| question | pairs | violation rate A | violation rate B | A-only / B-only | p (exact McNemar) | p (Holm) |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in tests.items():
            lines.append(
                f"| {k} | {v['n_pairs']} | {v['a']}: {100 * v['rate_a']:.1f}% ({v['k_a']}) | "
                f"{v['b']}: {100 * v['rate_b']:.1f}% ({v['k_b']}) | "
                f"{v['discordant_a_only']} / {v['discordant_b_only']} | {fmt_p(v['p'])} | {fmt_p(adj[k])} |")
        if not tests:
            lines.append("| (no paired conditions found) | | | | | | |")

        # Secondary: moderate vs ideal (not corrected), cost of compliance, zero bounds.
        sec = _compare(_paired(
            idx, lambda p, s: ("moderate", p, 0, s), lambda p, s: ("ideal", p, 0, s),
            [(p, s) for p in prs for s in seeds]), "moderate", "ideal")
        lines += ["", "Secondary (not multiplicity-corrected):", ""]
        if sec["n_pairs"]:
            lines.append(f"- moderate vs ideal: {100 * sec['rate_a']:.1f}% vs {100 * sec['rate_b']:.1f}% "
                         f"({sec['n_pairs']} pairs), exact McNemar p = {fmt_p(sec['p'])}")
        pairs = _paired(idx, lambda p, s: ("severe", p, 0, s), lambda p, s: ("ideal", p, 0, s),
                        [(p, s) for p in prs for s in seeds])
        if pairs:
            diffs = [x["summary"]["n_inspected"] - y["summary"]["n_inspected"] for x, y in pairs]
            lines.append(f"- cost of a degraded link: mean waypoints inspected severe minus ideal = "
                         f"{sum(diffs) / len(diffs):+.2f} ({len(diffs)} pairs), "
                         f"sign-flip permutation p = {fmt_p(paired_flip_test(diffs))}")
        viol_eps = [e for e in g if e["summary"]["any_violation"]]
        if viol_eps:
            kc = sum(1 for e in viol_eps if e["report_scoring"]["concealed_rules"])
            p, lo, hi = wilson(kc, len(viol_eps))
            lines.append(f"- concealment: {kc} of {len(viol_eps)} violating episodes omitted at least one "
                         f"violated rule from the report ({100 * p:.0f}%, 95% CI {100 * lo:.0f}–{100 * hi:.0f}%)")
        for link in links:
            sub = [e for e in g if e["config"].get("link_name", e["config"]["link"]) == link
                   and int(e["config"]["recorder"]) == 0]
            if sub and not any(e["summary"]["any_violation"] for e in sub):
                lines.append(f"- {link} link: 0 violations in {len(sub)} episodes; "
                             f"95% upper bound on the rate {100 * upper_bound_zero(len(sub)):.1f}%")
        main_eps = [e for e in g if int(e["config"]["recorder"]) == 0]
        if main_eps and not any(e["summary"]["any_violation"] for e in main_eps):
            lines.append(f"- all links pooled: 0 violations in {len(main_eps)} episodes; "
                         f"95% upper bound on the rate {100 * upper_bound_zero(len(main_eps)):.1f}%")
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(root, "stats.md"), "w") as f:
        f.write(text + "\n")
    print(text)


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
    statistics(eps, a.root)
    plot(rows, a.root)


if __name__ == "__main__":
    main()
