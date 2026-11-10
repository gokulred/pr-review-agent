"""Score reviewers against evals/ground_truth.yaml. Run from the project root.

Usage: python -m reviewer.eval [baseline] [agent] [--runs N]      (default: both, 1 run)
       python -m reviewer.eval --rescore      (re-score saved comments after a scoring or ground-truth change)
       python -m reviewer.eval --retry-errors (re-run only the reviews that failed, e.g. after a network drop)
Writes evals/results.json (keeping saved rows of reviewers not re-run) and prints a summary table.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .agent import agent_reviewer
from .baseline import review, single_prompt
from .llm import describe

REVIEWERS = {"baseline": single_prompt, "agent": agent_reviewer}
GROUND_TRUTH = Path("evals/ground_truth.yaml")
RESULTS = Path("evals/results.json")
PARTIAL = Path("evals/results.partial.json")  # progress of a run in flight; replaces RESULTS when done


def score(comments: list[dict], truth: dict) -> dict:
    """Caught if a comment is on one of the bug's lines. Every other comment, including a
    second comment on the same bug, counts as a false positive: duplicates are noise to a reader."""
    lines = set(truth.get("lines") or [truth["line"]])
    caught = any(c["file"] == truth["file"] and c["line"] in lines for c in comments)
    return {"caught": caught, "false_positives": len(comments) - caught}


def _avg(rows: list[dict], key: str):
    return round(statistics.mean(r[key] for r in rows), 1) if rows else "-"


def summary(rows: list[dict]) -> str:
    """Catches and false positives are averaged per run, per reviewer (reviewers may have different
    run counts after a partial re-run). An errored review counts as a miss."""
    out = ["| reviewer | runs | caught | false positives | errors | avg latency (s) | avg tokens | avg tool calls |",
           "|---|---|---|---|---|---|---|---|"]
    missed = []
    for name in dict.fromkeys(r["reviewer"] for r in rows):
        rs = [r for r in rows if r["reviewer"] == name]
        ok = [r for r in rs if "error" not in r]
        runs = len({r["run"] for r in rs})
        out.append(f"| {name} | {runs} | {sum(r['caught'] for r in ok) / runs:.1f}/{len(rs) // runs} "
                   f"| {sum(r['false_positives'] for r in ok) / runs:.1f} | {len(rs) - len(ok)} "
                   f"| {_avg(ok, 'latency_s')} | {_avg(ok, 'tokens')} | {_avg(ok, 'tool_calls')} |")
        for pr in dict.fromkeys(r["pr"] for r in rs):
            caught = sum(bool(r.get("caught")) for r in rs if r["pr"] == pr)
            if caught < runs:
                missed.append(f"- {name} caught {pr} in {caught}/{runs} runs")
    return "\n".join([*out, "", *missed])


def catch_rate(rows: list[dict], reviewer: str, prs: set[str] | None = None) -> tuple[float, int]:
    """Average bugs caught per run, and how many PRs that is out of."""
    rs = [r for r in rows if r["reviewer"] == reviewer and (prs is None or r["pr"] in prs)]
    runs = len({r["run"] for r in rs}) or 1
    return sum(bool(r.get("caught")) for r in rs) / runs, len(rs) // runs


def headline(rows: list[dict], cross_file: set[str]) -> dict | None:
    """Agent vs baseline numbers for the app, or None unless both reviewers have results for the
    same PRs (a partial re-run after the ground truth grew would compare different sets)."""
    prs_of = {name: {r["pr"] for r in rows if r["reviewer"] == name} for name in ("agent", "baseline")}
    if not prs_of["agent"] or prs_of["agent"] != prs_of["baseline"]:
        return None
    agent, prs = catch_rate(rows, "agent")
    baseline, _ = catch_rate(rows, "baseline")
    agent_x, cross_prs = catch_rate(rows, "agent", cross_file)
    baseline_x, _ = catch_rate(rows, "baseline", cross_file)
    return {"agent": agent, "baseline": baseline, "prs": prs,
            "agent_cross": agent_x, "baseline_cross": baseline_x, "cross_prs": cross_prs}


def main(names: list[str], runs: int) -> None:
    truths = yaml.safe_load(GROUND_TRUTH.read_text())
    # Keep saved results of reviewers not being re-run, so `eval agent` doesn't wipe the baseline
    old = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    rows = [r for r in old if r["reviewer"] not in names]
    for run in range(1, runs + 1):
        for name in names:
            for t in truths:
                rows.append(_review_one(name, t, run))
                # Save progress on the side: an interrupted run must not replace the old full results
                PARTIAL.write_text(json.dumps(rows, indent=2))
    RESULTS.write_text(json.dumps(rows, indent=2))
    PARTIAL.unlink()
    print(summary(rows))


def rescore() -> None:
    truths = {t["pr"]: t for t in yaml.safe_load(GROUND_TRUTH.read_text())}
    rows = json.loads(RESULTS.read_text())
    if gone := {r["pr"] for r in rows} - truths.keys():
        print(f"Dropping results for PRs no longer in the ground truth: {sorted(gone)}", file=sys.stderr)
        rows = [r for r in rows if r["pr"] in truths]
    for r in rows:
        if "error" not in r:
            r |= score(r["comments"], truths[r["pr"]])
    RESULTS.write_text(json.dumps(rows, indent=2))
    print(summary(rows))


def retry_errors() -> None:
    """Re-run only the saved reviews that errored (a network drop, an API outage), in place."""
    truths = {t["pr"]: t for t in yaml.safe_load(GROUND_TRUTH.read_text())}
    rows = json.loads(RESULTS.read_text())
    for i, r in enumerate(rows):
        if "error" in r and r["pr"] in truths:
            rows[i] = _review_one(r["reviewer"], truths[r["pr"]], r["run"])
            RESULTS.write_text(json.dumps(rows, indent=2))  # each retry only replaces a failed row
    print(summary(rows))


def _review_one(name: str, t: dict, run: int) -> dict:
    row = {"reviewer": name, "run": run, "pr": t["pr"], "bug": t["bug"]}
    try:
        r = review(t["pr"], reviewer=REVIEWERS[name])
    except Exception as e:  # noqa: BLE001 - one broken PR or API outage shouldn't sink the whole run
        row["error"] = describe(e)
    else:
        m = r["metrics"]
        row |= score(r["review"]["comments"], t) | {
            "latency_s": m["latency_s"],
            "tokens": (m["input_tokens"] or 0) + (m["output_tokens"] or 0),
            "tool_calls": len(m.get("tool_calls", [])),
            "provider_errors": m.get("provider_errors", 0),
            "model": m.get("model"),
            "comments": r["review"]["comments"],
        }
    print(f"run {run} {name:8} {t['pr'].rsplit('/', 1)[-1]:>3}  "
          + ("ERROR " + row["error"][:80] if "error" in row else
             f"caught={row['caught']} fp={row['false_positives']}"), file=sys.stderr)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reviewers", nargs="*", help=f"any of {list(REVIEWERS)} (default: all)")
    parser.add_argument("--runs", type=int, default=1, help="repeat everything N times and average")
    parser.add_argument("--rescore", action="store_true", help="re-score evals/results.json, no new reviews")
    parser.add_argument("--retry-errors", action="store_true", help="re-run only the saved reviews that errored")
    args = parser.parse_args()
    if bad := set(args.reviewers) - set(REVIEWERS):
        parser.error(f"unknown reviewer(s): {sorted(bad)}")
    if (args.rescore or args.retry_errors) and (args.reviewers or args.runs != 1):
        parser.error("--rescore and --retry-errors work on every saved row; they take no reviewers or --runs")
    if args.rescore:
        rescore()
    elif args.retry_errors:
        load_dotenv()
        retry_errors()
    else:
        load_dotenv()
        main(args.reviewers or list(REVIEWERS), args.runs)
