from unidiff import PatchSet


def _parse(f: dict):
    """unidiff view of one GitHub PR file, or None if GitHub sent no patch (binary or too big)."""
    if not f.get("patch"):
        return None
    name = f["filename"]
    # GitHub sends hunks only; unidiff needs the file headers
    return PatchSet(f"--- a/{name}\n+++ b/{name}\n{f['patch']}\n")[0]


def changed_lines(files: list[dict]) -> dict[str, set[int]]:
    """Map each file in a PR to the new-file line numbers its diff adds."""
    return {
        f["filename"]: {line.target_line_no for hunk in pf for line in hunk if line.is_added}
        for f in files
        if (pf := _parse(f))
    }


def numbered_diff(files: list[dict], max_lines: int | None = None) -> str:
    """Diff with new-file line numbers on every line, so the LLM can cite lines exactly.

    With max_lines, the diff is cut there and a note says how much was left out.
    """
    out = []
    for f in files:
        if not (pf := _parse(f)):
            continue
        out.append(f"### {f['filename']}")
        for hunk in pf:
            for line in hunk:
                num = line.target_line_no or ""  # removed lines have no new-file number
                out.append(f"{num:>5} {line.line_type}{line.value.rstrip()}")
    if max_lines is not None and len(out) > max_lines:
        out = [*out[:max_lines], f"(diff cut off: {len(out) - max_lines} more lines not shown)"]
    return "\n".join(out)


def filter_to_changed_lines(items, changed: dict[str, set[int]]) -> list:
    """Keep only items (anything with .file and .line) that sit on a line the PR added."""
    return [i for i in items if i.line in changed.get(i.file, ())]
