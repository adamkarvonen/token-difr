# %%
# Cell: dump selected files for LLM context
from pathlib import Path
from typing import Iterable

# Edit this: absolute or relative paths, glob patterns are fine
filepaths: list[str] = [
    # "local_eval.py",
    "tests/test_fireworks.py",
    "src/token_difr/api.py",
    "src/token_difr/openrouter_api.py",
    "src/token_difr/tinker_api.py",
]

DELIM = "─" * 80  # visual separator line
SHOW_HEADER = True  # set False if you don't want per-file headings


def iter_paths(paths: Iterable[str]) -> list[Path]:
    expanded = []
    for p in paths:
        # supports globs like "src/**/*.py"
        expanded.extend(Path().glob(p) if any(ch in p for ch in "*?[]") else [Path(p)])
    return [p.resolve() for p in expanded if p.is_file()]


def dump_files(paths: Iterable[str]):
    files = iter_paths(paths)
    for i, path in enumerate(files, 1):
        if SHOW_HEADER:
            print(f"{DELIM}\n# {path}  ({i}/{len(files)})\n{DELIM}")
        else:
            print(DELIM)
        print(path.read_text(encoding="utf-8", errors="replace").rstrip())  # avoid extra blank line
        print()  # spacer between files


dump_files(filepaths)

# %%
