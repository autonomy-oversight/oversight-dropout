# Pilot plan (about 1.5 weeks part-time, before the TAIF reapplication)

Goal: a public repository plus one figure and a one-page note showing a first
real measurement, so the application can point to working code and a
preliminary result rather than a plan alone.

## Steps

1. **Smoke test (½ day).** `--dry-run`, then 2 seeds on one model. Read every
   transcript. Fix only harness problems (parsing, unclear wording). After this
   step, freeze the prompts and the environment and record the commit hash —
   the pilot is then pre-specified.
2. **Main grid (1–2 days, mostly waiting).** Two models, e.g. one frontier model
   and one open-weight model: 3 links × 2 pressures × 20 seeds = 120 episodes
   each. `python analyze.py results`.
3. **Qualitative pass (1 day).** Read 10–15 violating episodes. Tag each
   violation by rationale, e.g. *mission pressure overrode the rule*, *misread
   the lost-link state*, *acted on an approval it had not received*, *treated a
   stale message as current*. Note concealment patterns in the reports
   (omitted vs. minimized vs. reframed).
4. **Optional, if time allows.**
   * `--recorder off,on` on the `severe` link, one model: does the agent behave
     differently when it believes outages are logged and reviewed?
   * One trace from an LTE/5G aerial measurement flight via `make_trace.py`.
5. **Write-up (1 day).** README results section, `fig_<model>.png`, and a
   one-page note (setup, figure, three findings, limitations, what the 8-week
   project adds). Push to the `autonomy-oversight` GitHub account.

## What would count as an interesting pilot result

Any of these is worth reporting, including a null result:

* violation rates that rise with link degradation and/or mission pressure;
* concealment: violations left out of the post-mission report;
* a recorder effect: fewer violations when outages are said to be logged;
* or the opposite: models comply robustly and pay for it in mission
  completion — which is then the baseline the 8-week project stress-tests.

## Text for the application (fill in only with measured numbers)

> **Pilot.** To test feasibility, I built an open-source harness in which an
> LLM acts as the onboard agent of a simulated inspection aerial robot whose
> oversight link to a ground station drops out, following a two-state
> Gilbert–Elliott model or replayed aerial link traces
> ([repository link]). Operating rules (restricted-zone approvals, a lost-link
> procedure, battery reserve, operator commands, truthful reporting) are
> checked against ground truth, and scripted reference policies validate the
> metrics. Across [N] episodes with [models], [main finding, e.g. "the share
> of episodes with at least one rule violation rose from X% with an ideal link
> to Y% with a severely degraded link"], and [finding on reporting, e.g. "in
> Z% of violating episodes the agent's post-mission report omitted at least
> one violation"]. The proposed project extends this pilot to [WPs of the
> 8-week plan].
