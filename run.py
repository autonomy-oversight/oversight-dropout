#!/usr/bin/env python3
"""Run the oversight-dropout pilot grid.

Examples
--------
# Validate the harness with the scripted reference policies (no API needed):
python run.py --agent rule_follower --seeds 0-49
python run.py --agent mission_first --seeds 0-49

# LLM agent via OpenRouter (any OpenAI-compatible endpoint works):
export OPENROUTER_API_KEY=...
python run.py --agent llm --model <model-id> --seeds 0-49 --workers 6 --max-tokens 4000

# Print the first prompt without calling any API:
python run.py --agent llm --model x --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from oversight_dropout.agents import make_agent
from oversight_dropout.env import EpisodeConfig, MissionEnv
from oversight_dropout.prompts import REPORT_REQUEST, system_prompt


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_episode(agent, cfg: EpisodeConfig) -> dict:
    env = MissionEnv(cfg)
    messages = [{"role": "system", "content": system_prompt(cfg.pressure, cfg.recorder)},
                {"role": "user", "content": env.observation()}]
    steps = []
    while True:
        obs_t, lost, batt = env.t, env.contact_lost_s(), env.battery
        n_viol_before = len(env.violations)
        raw, action = agent.act(env, messages)
        done = env.step(action)
        steps.append({
            "t": obs_t, "contact_lost_s": lost, "battery": round(batt, 2),
            "raw": raw, "action": action,
            "new_violations": [v.__dict__ for v in env.violations[n_viol_before:]],
            "result": env.last_result,
        })
        messages.append({"role": "assistant", "content": raw})
        if done:
            break
        messages.append({"role": "user", "content": env.observation()})

    end_note = ("Landed at HOME." if env.landed else
                "Flight terminated: battery depleted." if env.crashed else
                "Flight terminated: decision/time limit reached.")
    messages.append({"role": "user",
                     "content": f"{end_note}\nFinal status:\n{env.observation().rsplit(chr(10), 1)[0]}\n\n{REPORT_REQUEST}"})
    raw_rep, report = agent.report(env, messages)
    messages.append({"role": "assistant", "content": raw_rep})

    return {
        "agent": agent.tag,
        "config": cfg.__dict__,
        "summary": env.summary(),
        "violations": [v.__dict__ for v in env.violations],
        "report": report,
        "report_raw": raw_rep,
        "report_scoring": env.score_report(report),
        "steps": steps,
        "trajectory": env.trajectory,
        "messages": messages,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="rule_follower", choices=["llm", "rule_follower", "mission_first"])
    ap.add_argument("--model", default=None, help="model id for --agent llm")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: OpenRouter)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--history-window", type=int, default=0,
                    help="keep only the last N turns in context (0 = full history)")
    ap.add_argument("--links", default="ideal,moderate,severe",
                    help="comma list of presets and/or trace:<file.csv>")
    ap.add_argument("--pressures", default="low,high")
    ap.add_argument("--recorder", default="off", help="off, on, or off,on")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--recall-prob", type=float, default=0.3)
    ap.add_argument("--approval-policy", default="approve", choices=["approve", "deny"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    args = ap.parse_args()

    if args.dry_run:
        cfg = EpisodeConfig(link="severe", pressure="high", seed=0)
        env = MissionEnv(cfg)
        print("=== SYSTEM ===\n" + system_prompt(cfg.pressure, cfg.recorder))
        print("=== FIRST USER MESSAGE ===\n" + env.observation())
        return

    kw = {}
    if args.agent == "llm":
        if not args.model:
            sys.exit("--model is required for --agent llm")
        kw = dict(model=args.model, base_url=args.base_url, temperature=args.temperature,
                  max_tokens=args.max_tokens, history_window=args.history_window)
    tag = make_agent(args.agent, **kw).tag

    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)
    recs = {"off": [False], "on": [True], "off,on": [False, True], "on,off": [False, True]}[args.recorder]

    jobs = []
    for link in args.links.split(","):
        for pressure in args.pressures.split(","):
            for rec in recs:
                for seed in parse_seeds(args.seeds):
                    link_name = os.path.basename(link).replace(".csv", "") if link.startswith("trace:") else link
                    fn = os.path.join(outdir, f"{link_name}__{pressure}__rec{int(rec)}__s{seed:03d}.json")
                    if os.path.exists(fn):
                        continue
                    jobs.append((fn, EpisodeConfig(link=link, pressure=pressure, seed=seed, recorder=rec,
                                                   recall_prob=args.recall_prob,
                                                   approval_policy=args.approval_policy)))
    print(f"{len(jobs)} episodes to run -> {outdir}")

    def work(fn, cfg):
        agent = make_agent(args.agent, **kw)   # one client per episode (thread safety)
        ep = run_episode(agent, cfg)
        ep["usage"] = agent.usage
        ep["config"]["link_name"] = os.path.basename(fn).split("__")[0]
        with open(fn, "w") as f:
            json.dump(ep, f, indent=1)
        s = ep["summary"]
        return f"{os.path.basename(fn)}: inspected={s['n_inspected']} viol={s['violations_by_rule']} " \
               f"concealed={ep['report_scoring']['concealed_rules']}"

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(work, fn, cfg): fn for fn, cfg in jobs}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception:
                failures += 1
                print(f"FAILED {futs[fut]}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
    print(f"done; {failures} failures (re-run the same command to retry them)")


if __name__ == "__main__":
    main()
