import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolFinding:
    tool: str
    file: str
    line: int
    code: str
    message: str


def _run(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        # check=False: callers read the exit code, since mypy exits 1 when it finds errors
        [sys.executable, "-m", *args], check=False, cwd=cwd, capture_output=True, text=True, timeout=120,
    )


def run_ruff(repo_dir: str, files: list[str]) -> list[ToolFinding]:
    # --isolated ignores the repo's config; we pick bug (F, B) and security (S) rules ourselves
    # "--" so a PR file named like "--config=x.py" is read as a file, not an option
    p = _run(["ruff", "check", "--isolated", "--select", "F,B,S",
              "--output-format", "json", "--exit-zero", "--", *files], repo_dir)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    root = Path(repo_dir).resolve()
    return [
        ToolFinding("ruff", Path(d["filename"]).resolve().relative_to(root).as_posix(),
                    d["location"]["row"], d["code"] or "", d["message"])
        for d in json.loads(p.stdout)
    ]


def run_mypy(repo_dir: str, files: list[str]) -> list[ToolFinding]:
    # Empty --config-file skips the repo's config, which could load arbitrary mypy plugins.
    # "--" stops a PR file named "--config-file=x.py" from switching it back on.
    p = _run(["mypy", "--config-file=", "--ignore-missing-imports", "--follow-imports=silent",
              "--no-error-summary", "-O", "json", "--", *files], repo_dir)
    # Exit 1 = type errors, 2 = syntax errors or a crash. Syntax errors still come as JSON on stdout.
    if p.returncode > 1 and not p.stdout.strip():
        raise RuntimeError(p.stderr.strip())
    return [
        ToolFinding("mypy", Path(d["file"]).as_posix(), d["line"], d["code"] or "", d["message"])
        for d in map(json.loads, p.stdout.splitlines())
    ]


def run_tools(repo_dir: str, files: list[str]) -> tuple[list[ToolFinding], list[str]]:
    """Run every tool; a broken tool becomes an error string instead of failing the review."""
    findings, errors = [], []
    for tool in (run_ruff, run_mypy):
        try:
            findings += tool(repo_dir, files)
        except (subprocess.SubprocessError, OSError, RuntimeError, ValueError, KeyError) as e:
            errors.append(f"{tool.__name__}: {e}")
    return findings, errors
