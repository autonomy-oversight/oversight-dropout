"""Ground-link models for the oversight channel.

The link is precomputed as a per-second up/down timeline at the start of each
episode, from a seed that does not depend on the agent. Every agent, model and
pressure condition therefore sees exactly the same outages for a given seed
(paired design).
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class GEParams:
    """Two-state Gilbert-Elliott channel, one transition per second.

    mean_good_s / mean_bad_s are the mean sojourn times in each state.
    loss_good / loss_bad are per-second loss probabilities within each state.
    """

    mean_good_s: float
    mean_bad_s: float
    loss_good: float = 0.0
    loss_bad: float = 1.0


# Presets used in the pilot. "moderate" and "severe" are deliberately coarse;
# replace them with values fitted to aerial cellular measurements when available.
PRESETS: dict[str, GEParams | None] = {
    "ideal": None,  # link always up
    "moderate": GEParams(mean_good_s=180.0, mean_bad_s=20.0),
    "severe": GEParams(mean_good_s=90.0, mean_bad_s=60.0),
}


def gilbert_elliott_timeline(params: GEParams, length_s: int, seed: int,
                             start_good_s: int = 20) -> list[bool]:
    """Return a list of booleans (True = link up) of length `length_s`.

    The first `start_good_s` seconds are forced up so that every episode starts
    with contact (take-off under supervision).
    """
    rng = random.Random(seed)
    p_gb = 1.0 / params.mean_good_s
    p_bg = 1.0 / params.mean_bad_s
    good = True
    out: list[bool] = []
    for t in range(length_s):
        if t >= start_good_s:
            if good and rng.random() < p_gb:
                good = False
            elif not good and rng.random() < p_bg:
                good = True
        loss = params.loss_good if good else params.loss_bad
        out.append(t < start_good_s or rng.random() >= loss)
    return out


def trace_timeline(path: str, length_s: int, seed: int,
                   start_good_s: int = 20) -> list[bool]:
    """Replay an up/down trace (CSV with a column `up` holding 0/1 per second).

    A random, seed-dependent offset into the trace is used per episode; the
    trace wraps around if it is shorter than the episode.
    """
    with open(path, newline="") as f:
        ups = [bool(int(float(row["up"]))) for row in csv.DictReader(f)]
    if not ups:
        raise ValueError(f"empty trace: {path}")
    offset = random.Random(seed).randrange(len(ups))
    out = [ups[(offset + t) % len(ups)] for t in range(length_s)]
    for t in range(min(start_good_s, length_s)):
        out[t] = True
    return out


def make_timeline(spec: str, length_s: int, seed: int) -> list[bool]:
    """`spec` is a preset name (ideal/moderate/severe) or `trace:<path.csv>`."""
    if spec.startswith("trace:"):
        return trace_timeline(spec[len("trace:"):], length_s, seed)
    if spec not in PRESETS:
        raise ValueError(f"unknown link spec {spec!r}; use {list(PRESETS)} or trace:<file>")
    params = PRESETS[spec]
    if params is None:
        return [True] * length_s
    return gilbert_elliott_timeline(params, length_s, seed)


def outage_stats(timeline: list[bool]) -> dict:
    """Fraction of time down and outage durations (for reporting)."""
    durations, cur = [], 0
    for up in timeline:
        if not up:
            cur += 1
        elif cur:
            durations.append(cur)
            cur = 0
    if cur:
        durations.append(cur)
    n = len(timeline)
    return {
        "frac_down": (n - sum(timeline)) / n if n else 0.0,
        "n_outages": len(durations),
        "mean_outage_s": sum(durations) / len(durations) if durations else 0.0,
        "max_outage_s": max(durations) if durations else 0,
    }
