# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub==1.29.0"]
# ///
"""Tabulate every Job B experiment in one place.

Runs locally, not as a Job. Pulls `runs/*/metrics.json` from the experiments
repo plus the promoted model's own metrics, and prints one row per run.

Runs are ranked by **recall at a matched false-accept budget**, not by recall at
a fixed threshold. Measured 2026-09-02: three models out of the same pipeline
put threshold 0.5 at 0.47, 0.93 and 1.31 FA/h. Comparing their recall there
compared three different operating points and ranked them by how their scores
happened to be calibrated. Recall at a matched budget is a point on each model's
own ROC curve, and those are comparable.

    uv run scripts/compare_runs.py
    uv run scripts/compare_runs.py --budget 0.2
    uv run scripts/compare_runs.py --on babble      # rank on one validation condition

Runs from 2026-09-02 evening on carry per-condition recall (`clean`, `env`,
`music`, `speech`, `babble` -- the same validation clips under one background
role each) and a stress grid (held-out clips at fixed SNRs). Those columns
show `-` for older runs, whose validation set had no speech in it and could
not see the failure they measure.
"""

from __future__ import annotations

import argparse
import json
import math

RECALL_NOISE = 0.03   # paired 95% resolution on 2,000 validation positives
RECALL_REAL = 0.05    # the plan's threshold for believing a single difference
BUDGETS = [0.2, 0.5, 1.0, 2.0]
CONDITIONS = ["clean", "env", "music", "speech", "babble"]
STRESS_CELLS = ["babble@15", "babble@10", "music@10"]


def poisson_interval(events: int) -> tuple[float, float]:
    """Wilson-Hilferty approximation to the exact 95% interval, so this needs no
    scipy. The point is to show that a handful of events cannot separate two
    runs, and that survives a per-cent-level approximation."""
    if events == 0:
        return 0.0, 3.69
    lo = events * (1 - 1 / (9 * events) - 1.96 / (3 * math.sqrt(events))) ** 3
    hi = (events + 1) * (1 - 1 / (9 * (events + 1)) + 1.96 / (3 * math.sqrt(events + 1))) ** 3
    return max(0.0, lo), hi


def recall_at_fa(curve: list[dict], budget: float):
    """Best recall reachable without exceeding `budget` false accepts per hour."""
    reachable = [row["recall"] for row in curve
                 if row["false_accepts_per_hour"] <= budget]
    return max(reachable) if reachable else None


def load_metrics(api, repo: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    files = [f.rfilename for f in api.repo_info(repo, repo_type="model").siblings]
    out = []
    for name in sorted(f for f in files if f.endswith("metrics.json")):
        data = json.load(open(hf_hub_download(repo, name, repo_type="model")))
        data["_repo"], data["_path"] = repo, name
        out.append(data)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="iamtokarev/hey-sonny-experiments")
    ap.add_argument("--baseline-repo", default="iamtokarev/hey-sonny")
    ap.add_argument("--budget", type=float, default=0.5,
                    help="FA/h budget the ranking is decided at")
    ap.add_argument("--skip", nargs="*", default=["rehearsal"],
                    help="substrings of run names to leave out")
    ap.add_argument("--on", default="mixed", choices=["mixed"] + CONDITIONS,
                    help="rank on the mixed validation set or on one condition")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    runs = []
    for repo in (args.baseline_repo, args.repo):
        try:
            runs.extend(load_metrics(api, repo))
        except Exception as exc:
            print(f"  ({repo}: {exc})")
    if not runs:
        print("no metrics found")
        return 1

    rows = []
    for m in runs:
        name = m.get("run_name") or m["_repo"].split("/")[-1]
        if any(s in name for s in args.skip):
            continue
        curve = m["curve"]
        profile = m.get("recall_at_fa") or {str(b): recall_at_fa(curve, b) for b in BUDGETS}
        half = next(r for r in curve if r["threshold"] == 0.5)
        rows.append({
            "name": name, "sel": m.get("selected", "-"), "profile": profile,
            "half": half, "hrs": m["val_set_hrs"],
            # Runs before 2026-09-02 read recall off the 11-point reporting
            # grid. Where a model's scores pile up near 1.0 that grid puts its
            # low-FA operating point between 0.95 and 0.99 and reports a recall
            # it does not have, so those rows are not comparable with exact ones.
            #
            # `metrics_version` says so outright from v2 on. Before that, infer
            # it: a grid-derived profile can only contain recalls that appear in
            # the curve, because that is where it read them. An exact profile
            # lands between rows. A coincidence mislabels a row and changes no
            # number.
            "exact": (m.get("metrics_version", 0) >= 2
                      or "recall_at_fa_exact" in m
                      or any(v is not None and v not in {r["recall"] for r in curve}
                             for v in profile.values())),
            "merged": (m.get("selection") or {}).get("merged", "-"),
            "conditions": m.get("recall_at_fa_by_condition") or {},
            "stress": ((m.get("stress") or {}).get(m.get("selected"), {}) or {}).get("rows", {}),
            "baseline": m.get("baseline"),
        })

    head = "{0:<20} {1:<12}".format("run", "selected")
    head += "".join("{0:>10}".format("r@FA" + str(b)) for b in BUDGETS)
    head += "{0:>10}{1:>9}{2:>16}{3:>7}".format("r@t.5", "FA@t.5", "FA 95% CI", "merged")
    print(head)
    print("-" * len(head))
    for r in rows:
        label = r["name"][:18] + ("" if r["exact"] else " ~")
        line = "{0:<20} {1:<12}".format(label, str(r["sel"])[:12])
        for b in BUDGETS:
            v = r["profile"].get(str(b))
            line += "{0:>10}".format("-" if v is None else "{0:.3f}".format(v))
        lo, hi = poisson_interval(r["half"]["events"])
        line += "{0:>10.3f}{1:>9.2f}".format(r["half"]["recall"],
                                             r["half"]["false_accepts_per_hour"])
        line += "{0:>16}{1:>7}".format(
            "{0:.2f}-{1:.2f}".format(lo / r["hrs"], hi / r["hrs"]), str(r["merged"]))
        print(line)

    key = str(args.budget)
    if any(r["conditions"] for r in rows):
        print("\nRecall at <= {0} FA/h by validation condition, and stress-grid clear "
              "fraction (share of held-out\nclips above the same threshold):".format(args.budget))
        head2 = "{0:<20}".format("run") + "".join("{0:>8}".format(c) for c in CONDITIONS)
        head2 += "  |" + "".join("{0:>10}".format(c) for c in STRESS_CELLS)
        print(head2)
        print("-" * len(head2))
        for r in rows:
            def cell(v):
                return "-" if v is None else "{0:.3f}".format(v)
            line = "{0:<20}".format(r["name"][:18])
            line += "".join("{0:>8}".format(cell((r["conditions"].get(c) or {}).get(key, {}).get("recall")))
                            for c in CONDITIONS)
            line += "  |" + "".join("{0:>10}".format(cell(((r["stress"].get(c) or {}).get("clear_frac") or {}).get(key)))
                                    for c in STRESS_CELLS)
            print(line)
            b = r["baseline"]
            if b:
                bc = b.get("recall_at_fa_by_condition") or {}
                line = "{0:<20}".format("  (its baseline)")
                line += "".join("{0:>8}".format(cell((bc.get(c) or {}).get(key, {}).get("recall")))
                                for c in CONDITIONS)
                print(line)
    if args.on != "mixed":
        for r in rows:
            v = (r["conditions"].get(args.on) or {}).get(key, {}).get("recall")
            r["profile"] = {key: v}
        print("\nRanking below is on the {0!r} condition.".format(args.on))
    if any(not r["exact"] for r in rows):
        print("\n~ = recall read off the 11-point grid, not the exact ROC. "
              "Not comparable with the rest;\n    re-run to replace.")
    # Rank everything. Excluding rows produced "No run reaches 0.5 FA/h",
    # which is worse than a marker on the rows that need one.
    ranked = sorted((r for r in rows if r["profile"].get(key) is not None),
                    key=lambda r: r["profile"][key], reverse=True)
    print()
    if not ranked:
        print("No run reaches {0} FA/h at any threshold.".format(args.budget))
        return 0
    best = ranked[0]
    print("Ranked at a matched budget of {0} FA/h:".format(args.budget))
    for i, r in enumerate(ranked, 1):
        print("  {0}. {1:<20} {2:.3f}{3}".format(
            i, r["name"], r["profile"][key], "" if r["exact"] else "  ~grid"))

    # Mean across budgets, because no single budget resolves these differences
    # and the low-FA columns are the least reliable: at 0.2 FA/h the threshold
    # is pinned by two events.
    print("\nMean recall across all budgets (each row's own ROC):")
    for r in sorted(rows, key=lambda r: -sum(
            v for v in r["profile"].values() if v is not None)):
        vals = [v for v in r["profile"].values() if v is not None]
        got = len(vals)
        print("  {0:<20} {1:.3f} over {2} budget(s){3}".format(
            r["name"], sum(vals) / got if got else 0.0, got,
            "" if r["exact"] else "  ~grid"))
    if len(ranked) > 1:
        delta = best["profile"][key] - ranked[1]["profile"][key]
        verdict = ("noise" if delta < RECALL_NOISE
                   else "real" if delta > RECALL_REAL else "inconclusive on its own")
        print("\nMargin of {0} over {1}: {2:+.3f} -- {3}".format(
            best["name"], ranked[1]["name"], delta, verdict))
        wins = sum(1 for b in BUDGETS
                   if best["profile"].get(str(b)) is not None
                   and ranked[1]["profile"].get(str(b)) is not None
                   and best["profile"][str(b)] > ranked[1]["profile"][str(b)])
        seen = sum(1 for b in BUDGETS
                   if best["profile"].get(str(b)) is not None
                   and ranked[1]["profile"].get(str(b)) is not None)
        print("It leads at {0} of {1} budgets. A margin near the noise floor that "
              "holds at every budget\nis stronger evidence than the same margin at "
              "one.".format(wins, seen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
