from __future__ import annotations

import argparse
import ast
import difflib
import io
import re
import tokenize
from pathlib import Path
from typing import Iterable


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
}

CODING_RE = re.compile(
    r"coding[:=]\s*[-\w.]+"
)


def iter_python_files(
    root: Path,
) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(
            part in EXCLUDED_DIRS
            for part in path.parts
        ):
            continue

        yield path


def is_string_expr(
    node: ast.stmt,
) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def add_string_range(
    ranges: list[tuple[int, int, int, int]],
    node: ast.Expr,
) -> None:
    value = node.value

    if not hasattr(value, "lineno"):
        return

    if not hasattr(value, "end_lineno"):
        return

    ranges.append(
        (
            int(value.lineno),
            int(value.col_offset),
            int(value.end_lineno),
            int(value.end_col_offset),
        )
    )


def collect_string_ranges(
    source: str,
    remove_standalone_strings: bool,
) -> list[tuple[int, int, int, int]]:
    tree = ast.parse(
        source
    )

    ranges: list[tuple[int, int, int, int]] = []

    for node in ast.walk(
        tree
    ):
        if isinstance(
            node,
            (
                ast.Module,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            body = getattr(
                node,
                "body",
                [],
            )

            if body and is_string_expr(
                body[0]
            ):
                add_string_range(
                    ranges,
                    body[0],
                )

        if remove_standalone_strings and isinstance(
            node,
            ast.Expr,
        ):
            if is_string_expr(
                node
            ):
                add_string_range(
                    ranges,
                    node,
                )

    return list(
        dict.fromkeys(
            ranges
        )
    )


def token_inside_range(
    token: tokenize.TokenInfo,
    ranges: list[tuple[int, int, int, int]],
) -> bool:
    start_line, start_col = token.start
    end_line, end_col = token.end

    for (
        range_start_line,
        range_start_col,
        range_end_line,
        range_end_col,
    ) in ranges:
        starts_after = (
            start_line > range_start_line
            or (
                start_line == range_start_line
                and start_col >= range_start_col
            )
        )

        ends_before = (
            end_line < range_end_line
            or (
                end_line == range_end_line
                and end_col <= range_end_col
            )
        )

        if starts_after and ends_before:
            return True

    return False


def keep_special_comment(
    token: tokenize.TokenInfo,
) -> bool:
    line_number = token.start[0]
    text = token.string

    if line_number == 1 and text.startswith(
        "#!"
    ):
        return True

    if line_number <= 2 and CODING_RE.search(
        text
    ):
        return True

    return False


def normalize_blank_lines(
    text: str,
    max_consecutive_blank_lines: int = 2,
) -> str:
    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    output: list[str] = []
    blank_count = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1

            if blank_count <= max_consecutive_blank_lines:
                output.append(
                    ""
                )
        else:
            blank_count = 0
            output.append(
                line
            )

    while output and output[-1] == "":
        output.pop()

    return "\n".join(
        output
    ) + "\n"


def clean_source(
    source: str,
    remove_standalone_strings: bool,
) -> str:
    string_ranges = collect_string_ranges(
        source=source,
        remove_standalone_strings=remove_standalone_strings,
    )

    tokens: list[tokenize.TokenInfo] = []

    reader = io.StringIO(
        source
    ).readline

    for token in tokenize.generate_tokens(
        reader
    ):
        if token.type == tokenize.COMMENT:
            if keep_special_comment(
                token
            ):
                tokens.append(
                    token
                )

            continue

        if token.type == tokenize.STRING and token_inside_range(
            token,
            string_ranges,
        ):
            continue

        tokens.append(
            token
        )

    cleaned = tokenize.untokenize(
        tokens
    )

    cleaned = normalize_blank_lines(
        cleaned
    )

    return cleaned


def read_python_file(
    path: Path,
) -> tuple[str, str]:
    with tokenize.open(
        path
    ) as file:
        source = file.read()
        encoding = file.encoding

    return source, encoding


def process_file(
    path: Path,
    write: bool,
    show_diff: bool,
    remove_standalone_strings: bool,
) -> bool:
    try:
        source, encoding = read_python_file(
            path
        )

        cleaned = clean_source(
            source=source,
            remove_standalone_strings=remove_standalone_strings,
        )
    except SyntaxError as error:
        print(
            f"[SKIP syntax error] {path}: {error}"
        )
        return False
    except tokenize.TokenError as error:
        print(
            f"[SKIP tokenize error] {path}: {error}"
        )
        return False

    if cleaned == source:
        return False

    print(
        f"[CLEAN] {path}"
    )

    if show_diff:
        diff = difflib.unified_diff(
            source.splitlines(
                keepends=True
            ),
            cleaned.splitlines(
                keepends=True
            ),
            fromfile=str(
                path
            ),
            tofile=str(
                path
            )
            + " cleaned",
        )

        print(
            "".join(
                diff
            )
        )

    if write:
        path.write_text(
            cleaned,
            encoding=encoding,
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove Python comments and docstring/comment blocks "
            "from .py files."
        )
    )

    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root. Default: current directory.",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually modify files. Without this flag it only reports changes.",
    )

    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show unified diff for every modified file.",
    )

    parser.add_argument(
        "--remove-standalone-strings",
        action="store_true",
        help=(
            "Remove every standalone string expression, not only official "
            "module/class/function docstrings. Useful for triple-quoted "
            "comment blocks."
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    ).resolve()

    if not root.exists():
        raise FileNotFoundError(
            root
        )

    changed = 0
    scanned = 0

    for path in iter_python_files(
        root
    ):
        scanned += 1

        if process_file(
            path=path,
            write=args.write,
            show_diff=args.diff,
            remove_standalone_strings=args.remove_standalone_strings,
        ):
            changed += 1

    print()
    print(
        "=" * 72
    )
    print(
        "Python comment cleanup finished"
    )
    print(
        "=" * 72
    )
    print(
        f"Scanned files : {scanned}"
    )
    print(
        f"Changed files : {changed}"
    )
    print(
        f"Write mode    : {args.write}"
    )


if __name__ == "__main__":
    main()
