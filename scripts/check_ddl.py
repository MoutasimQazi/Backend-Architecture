"""Guard against MySQL/MariaDB reserved words used as unquoted identifiers.

This class of bug does not show up in any Python test — the SQL is only a string
until it reaches the server, where it fails with a bare 1064. Three tables in
this schema were caught this way (`ocr_result.text`, `evidence_chunk.text`,
`evidence_document.year`), each of which would have silently failed to create
during a phpMyAdmin import and then surfaced as a 1146 at runtime.

    python scripts/check_ddl.py

Exits non-zero on a collision, so it can gate a deploy.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIGRATIONS = Path(__file__).resolve().parent.parent / "packages" / "storage" / "migrations"

# MySQL 8.0 / MariaDB reserved words that are plausible as table or column
# names. Not the full list — the full list is mostly SQL clause keywords no one
# would pick. These are the ones a schema author actually reaches for.
RESERVED = {
    "blob", "text", "year", "order", "group", "key", "condition", "range",
    "interval", "lines", "rank", "system", "usage", "check", "column", "table",
    "index", "primary", "foreign", "default", "values", "select", "insert",
    "update", "delete", "from", "where", "join", "left", "right", "inner",
    "outer", "union", "int", "integer", "float", "double", "decimal", "date",
    "datetime", "timestamp", "time", "char", "varchar", "binary", "varbinary",
    "long", "longtext", "mediumtext", "tinytext", "precision", "numeric",
    "read", "write", "lock", "option", "force", "ignore", "match", "against",
    "partition", "explain", "describe", "analyze", "optimize", "repair",
    "current_date", "current_time", "current_timestamp", "localtime",
    "asc", "desc", "distinct", "having", "limit", "offset", "into", "set",
    "null", "true", "false", "and", "or", "not", "in", "is", "like", "between",
    "exists", "case", "when", "then", "else", "end", "as", "on", "using",
    "cross", "natural", "straight_join", "for", "with", "recursive", "window",
    "over", "rows", "unbounded", "preceding", "following", "current", "row",
    "if", "elseif", "while", "loop", "repeat", "leave", "iterate", "return",
    "call", "declare", "cursor", "fetch", "close", "open", "handler",
    "signal", "resignal", "escape", "regexp", "rlike", "sounds", "div", "mod",
    "xor", "collate", "convert", "cast", "separator", "outfile", "dumpfile",
    "load", "replace", "high_priority", "low_priority", "delayed", "quick",
    "sql_calc_found_rows", "distinctrow", "straight", "before", "after",
}

_CREATE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+(`?)(\w+)\1", re.IGNORECASE)
_COLUMN = re.compile(
    r"^(`?)(\w+)\1\s+"
    r"(VARCHAR|CHAR|TEXT|MEDIUMTEXT|LONGTEXT|TINYTEXT|INT|BIGINT|SMALLINT|TINYINT|"
    r"DECIMAL|NUMERIC|FLOAT|DOUBLE|DATE|DATETIME|TIMESTAMP|TIME|YEAR|JSON|"
    r"VARBINARY|BINARY|BLOB|ENUM|BOOL)",
    re.IGNORECASE,
)


def check() -> list[str]:
    problems: list[str] = []

    for path in sorted(MIGRATIONS.glob("*.sql")):
        table = None
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue

            create = _CREATE.match(stripped)
            if create:
                quoted, table = bool(create.group(1)), create.group(2)
                if table.lower() in RESERVED and not quoted:
                    problems.append(
                        f"{path.name}:{lineno}  TABLE `{table}` is a reserved word "
                        f"— rename it (quoting works in DDL but every query must quote it too)"
                    )
                continue

            column = _COLUMN.match(stripped)
            if column:
                quoted, name = bool(column.group(1)), column.group(2)
                if name.lower() in RESERVED and not quoted:
                    problems.append(
                        f"{path.name}:{lineno}  COLUMN {table}.{name} is a reserved word — rename it"
                    )

    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"{len(problems)} reserved-word collision(s):\n")
        for problem in problems:
            print(f"  {problem}")
        print("\nThese fail at CREATE TABLE time with ERROR 1064, not in any Python test.")
        return 1

    tables = sum(
        len(_CREATE.findall(p.read_text(encoding="utf-8")))
        for p in MIGRATIONS.glob("*.sql")
    )
    print(f"no reserved-word collisions across {tables} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
