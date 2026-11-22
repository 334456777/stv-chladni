"""Lists the comments of stv's own source files in numbered chunk files, to be read one after another.

Usage: list_comments.py <out dir> [<chunk KiB>]

The project root is found from this script's location (.claude/skills/stv-trim-prose/scripts/), so the script runs
the same from any working directory. Files are found by walking the project; the build output,
vendored and output folders are left out (see SKIPPED_DIRS). Markdown is not listed, since those files are
read whole.

Each chunk (64 KiB by default) holds a `== <path>` line for each file, then one line for each comment: the line it
starts on and its text, the lines of a comment joined into one. Line, doc and block comments, Python docstrings and
comments after code are listed. The chunk files already in <out dir> are removed first.

The parsing goes line by line and is not a tokenizer: a comment marker inside a string can slip in, and a comment
after code is missed when the code before it holds an odd number of quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

SUFFIXES = {".rs", ".wgsl", ".toml", ".py", ".ps1", ".sh", ".yml", ".yaml"}
SKIPPED_DIRS = {"target", "vendor", "out", ".git", "node_modules", "__pycache__"}
HASH = {".py", ".sh", ".ps1", ".yml", ".yaml", ".toml"}
QUOTES = {".rs": '"', ".wgsl": ""}
BLOCKS = {".py": [('"""', '"""')], ".ps1": [("<#", "#>")]}


def after_code(line: str, marker: str, quotes: str) -> str | None:
    """The text of a comment that follows code on this line, if there is one."""
    start = 0
    while (i := line.find(marker, start)) > 0:
        before = line[:i]
        if before[-1].isspace() and before.strip() and all(before.count(q) % 2 == 0 for q in quotes):
            return line[i + len(marker):].strip()
        start = i + 1
    return None


def comments(path: Path, text: str) -> list[tuple[int, str]]:
    """Every comment of one file: the line it starts on and its text."""
    ext = path.suffix
    hashed = ext in HASH
    marker = "#" if hashed else "//"
    quotes = QUOTES.get(ext, "\"'")
    blocks = BLOCKS.get(ext, [] if hashed else [("/*", "*/")])
    found: list[tuple[int, str]] = []
    parts: list[str] = []
    first = 0
    closer = None

    def flush() -> None:
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            found.append((first, joined))
        parts.clear()

    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if closer:
            end = stripped.find(closer)
            body = stripped if end < 0 else stripped[:end]
            parts.append(body.lstrip("*").strip())
            if end >= 0:
                closer = None
                flush()
            continue
        opened = next(((o, c) for o, c in blocks if stripped.startswith(o) or stripped.startswith("r" + o)), None)
        if opened:
            flush()
            opener, end_marker = opened
            first = number
            body = stripped[stripped.find(opener) + len(opener):].lstrip("*!")
            end = body.find(end_marker)
            parts.append(body if end < 0 else body[:end])
            if end < 0:
                closer = end_marker
            else:
                flush()
            continue
        is_line_comment = stripped.startswith(marker) and not (hashed and stripped[1:2] in ("!", "["))
        if is_line_comment:
            if not parts:
                first = number
            parts.append(stripped[len(marker):].lstrip("/!").strip())
            continue
        flush()
        trailing = after_code(line, marker, quotes)
        if trailing:
            found.append((number, trailing))
    flush()
    return found


def source_files(root: Path) -> list[Path]:
    """The project's own source files, in a stable order."""
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIPPED_DIRS for part in rel.parts[:-1]):
            continue
        if path.is_file() and path.suffix in SUFFIXES:
            files.append(path)
    return files


def main() -> None:
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    root = Path(__file__).resolve().parents[4]
    out = Path(sys.argv[1])
    if not out.is_absolute():
        out = Path.cwd() / out
    limit = int(sys.argv[2] if len(sys.argv) == 3 else 64) * 1024

    files = source_files(root)
    chunks: list[str] = []
    rows: list[str] = []
    size = 0
    total = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        header = f"== {path.relative_to(root).as_posix()}\n"
        pending = header
        for number, comment in comments(path, text):
            row = f"{number}: {comment}\n"
            if rows and size + len(((pending or "") + row).encode()) > limit:
                chunks.append("".join(rows))
                rows, size, pending = [], 0, header
            if pending:
                rows.append(pending)
                size += len(pending.encode())
                pending = None
            rows.append(row)
            size += len(row.encode())
            total += 1
    if rows:
        chunks.append("".join(rows))

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("chunk-*.txt"):
        old.unlink()
    for index, chunk in enumerate(chunks):
        (out / f"chunk-{index:02}.txt").write_text(chunk, encoding="utf-8", newline="\n")
    print(f"{total} comments from {len(files)} files in {len(chunks)} chunks under {out}")


if __name__ == "__main__":
    main()
