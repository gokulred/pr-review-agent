"""Streamlit UI for the PR review agent.

Run: streamlit run app.py
Env: ALLOWED_REPOS=owner/repo,owner/repo2 limits which repos can be reviewed. It is also required
for posting, so a public deploy can't post as your token to arbitrary repos. With neither it nor
ALLOW_ANY_REPO=1 set, reviews are disabled.
"""
import json
import os
import threading
from pathlib import Path

import httpx
import streamlit as st
import yaml
from dotenv import load_dotenv

from reviewer.baseline import review
from reviewer.eval import REVIEWERS, headline
from reviewer.github_client import allowed_repos, parse_pr_url, post_review
from reviewer.llm import describe

load_dotenv()
EVALS = Path(__file__).parent / "evals"
RESULTS, GROUND_TRUTH = EVALS / "results.json", EVALS / "ground_truth.yaml"
BADGE = {"bug": ":orange-badge[bug]", "security": ":red-badge[security]"}

st.set_page_config(page_title="PR Review Agent", page_icon=":material/rate_review:", layout="wide")


@st.cache_resource
def review_slots() -> threading.BoundedSemaphore:
    """Shared across all browser sessions: free-tier LLM rate limits can't take more than 2 at once."""
    return threading.BoundedSemaphore(2)


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# The file's modification time is part of the cache key, so a new eval or --rescore shows up
# without restarting the app.
@st.cache_data
def eval_rows(modified: float) -> list[dict]:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else []


@st.cache_data
def cross_file_prs(modified: float) -> set[str]:
    truths = yaml.safe_load(GROUND_TRUTH.read_text()) if GROUND_TRUTH.exists() else []
    return {t["pr"] for t in truths if t.get("kind") == "cross-file"}


def code_span(text: str) -> str:
    """Untrusted text (file paths from the PR) as inline code that can't break out of the backticks."""
    return "`" + text.replace("`", "'") + "`"


def repo_of(url: str) -> str | None:
    try:
        owner, repo, _ = parse_pr_url(url)
    except ValueError:
        return None
    return f"{owner}/{repo}".lower()


def run_review(name: str, url: str) -> dict | None:
    with st.status(f"{name.title()} is reviewing…", expanded=True) as status:
        try:
            # status.text, not status.write: progress includes the PR title, which the PR author
            # controls and could fill with Markdown links or tracking images
            result = review(url, REVIEWERS[name], progress=status.text)
        except Exception as e:  # noqa: BLE001 - show any failure in the UI
            status.update(label=f"{name.title()} failed", state="error")
            st.error(describe(e))
            return None
        n = len(result["review"]["comments"])
        status.update(label=f"{name.title()}: {n} comment{'s' * (n != 1)}", state="complete", expanded=False)
    return result


def show_result(name: str, res: dict) -> None:
    m = res["metrics"]
    st.subheader(name.title())
    cols = st.columns(3)
    cols[0].metric("Latency", f"{m['latency_s']} s")
    cols[1].metric("Tokens", f"{(m['input_tokens'] or 0) + (m['output_tokens'] or 0):,}")
    cols[2].metric("Tool calls", len(m.get("tool_calls", [])))
    st.text(res["review"]["summary"])  # LLM output shaped by the PR's text: show it as plain text

    comments = res["review"]["comments"]
    if not comments:
        st.success("No bugs or security issues found.")
    for c in comments:
        with st.container(border=True):
            where = code_span(f"{c['file']}:{c['line']}")
            st.markdown(f"{where} {BADGE[c['category']]} :gray-badge[{c['severity']}]")
            st.text(c["message"])
            if c.get("suggestion"):
                st.text(f"Suggestion: {c['suggestion']}")

    with st.expander("Diff"):
        for path, patch in res["pr"]["patches"].items():
            st.markdown(code_span(path))
            st.code(patch or "(no text diff)", language="diff")
    if m.get("tool_calls"):
        with st.expander("Agent tool calls"):
            st.code("\n".join(m["tool_calls"]), language="text")

    if not (comments and repo_of(res["pr"]["url"]) in allowed_repos()):
        return
    posted_key = f"posted-{name}-{res['pr']['url']}-{res['pr']['head_sha']}"  # one post per review
    if link := st.session_state.get(posted_key):
        st.success(f"Posted: [view on GitHub]({link})")
    elif st.button("Post comments to GitHub", key=f"post-{name}", icon=":material/send:"):
        try:
            st.session_state[posted_key] = post_review(res["pr"]["url"], res["pr"]["head_sha"],
                                                       res["review"]["summary"], comments)
            st.rerun()  # redraw with the link instead of the button
        except httpx.HTTPError as e:
            st.error(f"GitHub rejected the review: {str(e)[:300]}")


# ---------- page ----------
st.title("PR Review Agent")
st.caption("A LangGraph agent that reviews Python pull requests for bugs and security issues. "
           "It runs ruff and mypy, then searches the repo for callers of the code a PR changes.")

rows = eval_rows(mtime(RESULTS))
if h := headline(rows, cross_file_prs(mtime(GROUND_TRUTH))):  # None unless both reviewers have results
    c1, c2, c3 = st.columns(3)
    c1.metric("Agent: bugs caught", f"{h['agent']:.1f} / {h['prs']}", f"{h['agent'] - h['baseline']:+.1f} vs baseline")
    c2.metric("Baseline: bugs caught", f"{h['baseline']:.1f} / {h['prs']}")
    c3.metric(f"Cross-file bugs (of {h['cross_prs']}): agent vs baseline",
              f"{h['agent_cross']:.1f} vs {h['baseline_cross']:.1f}")

tab_review, tab_eval = st.tabs(["Review a PR", "Eval results"])

with tab_review:
    allowed = sorted(allowed_repos())
    # Fail closed: a deploy that forgot ALLOWED_REPOS would otherwise clone any repo on your LLM keys
    any_repo = os.environ.get("ALLOW_ANY_REPO") == "1"
    if allowed:
        st.info(f"This demo reviews PRs in: {', '.join(allowed)}", icon=":material/info:")
    elif not any_repo:
        st.warning("Set ALLOWED_REPOS=owner/repo (or ALLOW_ANY_REPO=1 for local use) to enable reviews.",
                   icon=":material/lock:")
    with st.form("review"):
        url = st.text_input("Pull request URL",
                            placeholder=f"https://github.com/{allowed[0] if allowed else 'owner/repo'}/pull/1")
        mode = st.segmented_control("Reviewer", ["Agent", "Baseline", "Compare both"], default="Agent")
        submitted = st.form_submit_button("Review", type="primary", icon=":material/play_arrow:")

    if submitted:
        repo = repo_of(url)
        names = ["baseline", "agent"] if mode == "Compare both" else [(mode or "Agent").lower()]
        if not repo:
            st.error("That doesn't look like a GitHub pull request URL.")
        elif not allowed and not any_repo:
            st.error("Reviews are disabled until ALLOWED_REPOS or ALLOW_ANY_REPO=1 is set.")
        elif allowed and repo not in allowed:
            st.error(f"This demo only reviews: {', '.join(allowed)}")
        elif not review_slots().acquire(blocking=False):
            st.warning("Busy with other reviews. Try again in a minute.")
        else:
            try:
                results = {}
                for name in names:
                    if res := run_review(name, url):
                        results[name] = res
                st.session_state.results = results
            finally:
                review_slots().release()

    results = st.session_state.get("results", {})
    for col, (name, res) in zip(st.columns(len(results) or 1), results.items(), strict=False):
        with col:
            show_result(name, res)

with tab_eval:
    if not rows:
        st.write("No eval results yet. Run `python -m reviewer.eval --runs 3`.")
    else:
        runs = len({r["run"] for r in rows})
        st.write(f"Each PR has one planted bug. Every PR was reviewed {runs} times by each reviewer. "
                 "A catch is a comment on one of the lines the planted bug changed.")
        table = {}
        for r in rows:
            entry = table.setdefault(r["pr"], {"PR": "#" + r["pr"].rsplit("/", 1)[-1], "Planted bug": r["bug"],
                                               "Baseline": 0, "Agent": 0})
            entry[r["reviewer"].title()] += bool(r.get("caught"))
        st.dataframe(
            list(table.values()), hide_index=True, width="stretch",
            column_config={
                "Baseline": st.column_config.ProgressColumn(f"Baseline (of {runs})", min_value=0, max_value=runs,
                                                            format="%d"),
                "Agent": st.column_config.ProgressColumn(f"Agent (of {runs})", min_value=0, max_value=runs,
                                                         format="%d"),
            },
        )
