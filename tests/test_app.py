from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def test_app_renders_eval_headline_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[0] == "Agent: bugs caught" and "Cross-file bugs" in labels[2]


def test_bad_url_shows_error_without_calling_github(monkeypatch):
    monkeypatch.delenv("ALLOWED_REPOS", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("not a url")
    at.button[0].click().run()
    assert at.error[0].value.startswith("That doesn't look like")


def test_reviews_disabled_without_allowed_repos(monkeypatch):
    monkeypatch.setenv("ALLOWED_REPOS", "")  # set (so .env can't fill it) but empty
    monkeypatch.delenv("ALLOW_ANY_REPO", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("https://github.com/someone/big-repo/pull/1")
    at.button[0].click().run()
    assert at.error[0].value.startswith("Reviews are disabled")


def test_allow_any_repo_runs_a_review(monkeypatch):
    import reviewer.baseline

    def fake_review(url, reviewer, progress):
        progress("Fetched PR")
        return {"pr": {"url": url, "head_sha": "abc", "patches": {}},
                "review": {"summary": "Looks fine.", "comments": []},
                "metrics": {"latency_s": 1.0, "input_tokens": 5, "output_tokens": 5}}

    monkeypatch.setattr(reviewer.baseline, "review", fake_review)
    monkeypatch.setenv("ALLOWED_REPOS", "")
    monkeypatch.setenv("ALLOW_ANY_REPO", "1")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("https://github.com/someone/any-repo/pull/3")
    at.button[0].click().run()
    assert not at.error and not at.exception
    assert "Looks fine." in [t.value for t in at.text]


def test_post_button_posts_once_then_shows_the_link(monkeypatch):
    import reviewer.baseline
    import reviewer.github_client

    comment = {"file": "a.py", "line": 1, "category": "bug", "severity": "high", "message": "m", "suggestion": ""}
    monkeypatch.setattr(reviewer.baseline, "review", lambda url, reviewer, progress: {
        "pr": {"url": url, "head_sha": "abc", "patches": {}},
        "review": {"summary": "s", "comments": [comment]},
        "metrics": {"latency_s": 1.0, "input_tokens": 1, "output_tokens": 1}})
    posts = []
    monkeypatch.setattr(reviewer.github_client, "post_review",
                        lambda *args: posts.append(args) or "https://github.com/me/sandbox/pull/1#review")
    monkeypatch.setenv("ALLOWED_REPOS", "me/sandbox")
    monkeypatch.setenv("ENABLE_POSTING", "1")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("https://github.com/me/sandbox/pull/1")
    at.button[0].click().run()
    post = next(b for b in at.button if b.label == "Post comments to GitHub")
    post.click().run()
    assert len(posts) == 1
    assert not any(b.label == "Post comments to GitHub" for b in at.button)  # replaced by the link
    assert any("view on GitHub" in s.value for s in at.success)


def test_post_button_hidden_unless_posting_is_enabled(monkeypatch):
    import reviewer.baseline

    comment = {"file": "a.py", "line": 1, "category": "bug", "severity": "high", "message": "m", "suggestion": ""}
    monkeypatch.setattr(reviewer.baseline, "review", lambda url, reviewer, progress: {
        "pr": {"url": url, "head_sha": "abc", "patches": {}},
        "review": {"summary": "s", "comments": [comment]},
        "metrics": {"latency_s": 1.0, "input_tokens": 1, "output_tokens": 1}})
    monkeypatch.setenv("ALLOWED_REPOS", "me/sandbox")
    monkeypatch.setenv("ENABLE_POSTING", "")  # set but off, so a local .env can't turn it on
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input[0].input("https://github.com/me/sandbox/pull/1")
    at.button[0].click().run()
    assert not at.exception
    assert not any(b.label == "Post comments to GitHub" for b in at.button)
