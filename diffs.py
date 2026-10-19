"""Parse model replies: a full program, or OpenEvolve-style SEARCH/REPLACE blocks.

    <<<<<<< SEARCH
    exact lines from the parent
    =======
    replacement lines
    >>>>>>> REPLACE

Diffs are applied strictly, in order: every SEARCH text must occur exactly once in
the program as it stands after the previous blocks. No fuzzy or whitespace-tolerant
matching. Any problem rejects the whole edit with a message saying which block failed.
"""

import re

_BLOCK = re.compile(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)
_MARKERS = ("<<<<<<< SEARCH", ">>>>>>> REPLACE")


class DiffError(Exception):
    pass


def has_diff(text):
    return any(m in text for m in _MARKERS)


def parse_blocks(text):
    blocks = _BLOCK.findall(text)
    opened = text.count("<<<<<<< SEARCH")
    closed = text.count(">>>>>>> REPLACE")
    if opened != closed or opened != len(blocks):
        raise DiffError(f"malformed SEARCH/REPLACE markup: {opened} SEARCH markers, {closed} REPLACE "
                        f"markers, {len(blocks)} well-formed blocks")
    return blocks


def _line_matches(src, search):
    """Start offsets where `search` occurs as whole lines (no partial-line matches)."""
    out, i = [], src.find(search)
    while i >= 0:
        end = i + len(search)
        if (i == 0 or src[i - 1] == "\n") and (end == len(src) or src[end] == "\n"):
            out.append(i)
        i = src.find(search, i + 1)
    return out


def apply_diff(parent, text):
    """Apply every block in `text` to `parent`. Returns the new program."""
    blocks = parse_blocks(text)
    if not blocks:
        raise DiffError("no SEARCH/REPLACE blocks found")
    src = parent
    for i, (search, replace) in enumerate(blocks, 1):
        if not search.strip():
            raise DiffError(f"block {i}: SEARCH section is empty")
        hits = _line_matches(src, search)
        count = len(hits)
        if count == 0:
            first = search.strip().split("\n")[0][:80]
            raise DiffError(f"block {i}: SEARCH text not found in the parent (first line: {first!r}). "
                            "SEARCH must copy the parent's lines exactly, including indentation.")
        if count > 1:
            raise DiffError(f"block {i}: SEARCH text occurs {count} times; include more context "
                            "so it matches exactly once")
        src = src[: hits[0]] + replace + src[hits[0] + len(search):]
    if src == parent:
        raise DiffError("the edit leaves the program unchanged")
    return src


def extract_full_program(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    blocks = [b for b in blocks if "def build_circuit" in b]
    if blocks:
        return max(blocks, key=len)
    return None


def parse_reply(parent, text):
    """Return (source, edit_format). edit_format is "diff" or "full".
    Raises DiffError with model-readable feedback when nothing usable is found."""
    if has_diff(text):
        return apply_diff(parent, text), "diff"
    full = extract_full_program(text)
    if full is not None:
        return full, "full"
    raise DiffError("no SEARCH/REPLACE blocks and no ```python block containing def build_circuit")
