import os
import re
import subprocess

import httpx

PR_URL = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)/?")


def allowed_repos() -> set[str]:
    """Repos ("owner/repo", lowercase) from ALLOWED_REPOS. The app treats an empty set as
    "reviews disabled" (fail closed) unless ALLOW_ANY_REPO=1."""
    return {r.strip().lower() for r in os.environ.get("ALLOWED_REPOS", "").split(",") if r.strip()}


def parse_pr_url(url: str) -> tuple[str, str, int]:
    m = PR_URL.fullmatch(url.strip())
    if not m:
        raise ValueError(f"Not a GitHub PR URL: {url}")
    return m[1], m[2], int(m[3])


def _get(client: httpx.Client, path: str, **kwargs):
    r = client.get(path, **kwargs)
    if r.status_code in (403, 429) and r.headers.get("x-ratelimit-remaining") == "0":
        raise RuntimeError(
            "GitHub rate limit hit. Set GITHUB_TOKEN or wait until epoch "
            + r.headers.get("x-ratelimit-reset", "?")
        )
    r.raise_for_status()
    return r.json()


def _client() -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url="https://api.github.com", headers=headers, timeout=30)


def post_review(url: str, head_sha: str, summary: str, comments: list[dict]) -> str:
    """Post the review on the PR as inline comments. Needs a GITHUB_TOKEN with pull request write access.

    Pinned to head_sha, the commit that was reviewed, so line numbers stay valid if the PR moved on.
    Returns the review's URL.
    """
    owner, repo, number = parse_pr_url(url)
    body = {
        "commit_id": head_sha,
        "event": "COMMENT",
        "body": f"**AI review:** {summary}",
        "comments": [
            {"path": c["file"], "line": c["line"], "side": "RIGHT",
             "body": f"**{c['category']} ({c['severity']})**: {c['message']}"
                     + (f"\n\nSuggestion: {c['suggestion']}" if c.get("suggestion") else "")}
            for c in comments
        ],
    }
    with _client() as c:
        r = c.post(f"/repos/{owner}/{repo}/pulls/{number}/reviews", json=body)
        r.raise_for_status()
        return r.json()["html_url"]


def fetch_pr(url: str) -> dict:
    owner, repo, number = parse_pr_url(url)
    base = f"/repos/{owner}/{repo}/pulls/{number}"
    with _client() as c:
        pr = _get(c, base)
        files, page = [], 1
        while batch := _get(c, f"{base}/files", params={"per_page": 100, "page": page}):
            files += batch
            page += 1
    return {
        "owner": owner,
        "repo": repo,
        "number": number,
        "title": pr["title"],
        "body": pr["body"] or "",
        "head_sha": pr["head"]["sha"],
        "files": files,
    }


def checkout(owner: str, repo: str, number: int, dest: str) -> None:
    """Shallow-clone the PR head into dest. pull/N/head works for PRs from forks too."""
    def git(*args):
        subprocess.run(["git", *args], cwd=dest, check=True, capture_output=True, timeout=120)

    git("init", "-q")
    git("fetch", "-q", "--depth", "1", f"https://github.com/{owner}/{repo}.git", f"pull/{number}/head")
    git("checkout", "-q", "FETCH_HEAD")
