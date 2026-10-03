"""The environment is read only where a process starts; everything below is handed what it needs from it."""

import ast
from pathlib import Path

import hands

SOURCE = Path(hands.__file__).parent

# Where a process starts: the daemon and its CLI, and the two modules Claude Code and a skill run as processes of
# their own. The settings come from the home's config.toml, and only secrets and the home's own path from here.
STARTS = ("hands.daemon", "hands.sessions.shim", "hands.sessions.attention")

READS = {"environ", "environb", "getenv", "getenvb", "putenv", "unsetenv"}


def _module(path: Path) -> str:
    return ".".join(("hands", *path.relative_to(SOURCE).with_suffix("").parts))


def _reads(tree: ast.AST) -> list[int]:
    """The lines that reach the process environment through os."""
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in READS and isinstance(node.value, ast.Name) and node.value.id == "os")
        or (isinstance(node, ast.ImportFrom) and node.module == "os" and any(alias.name in READS for alias in node.names))
    )


def test_nothing_below_where_a_process_starts_reads_the_environment() -> None:
    read = {
        f"{_module(path)}:{line}"
        for path in SOURCE.rglob("*.py")
        if not any(_module(path) == start or _module(path).startswith(f"{start}.") for start in STARTS)
        for line in _reads(ast.parse(path.read_text()))
    }
    assert read == set()


def test_the_check_sees_a_read() -> None:
    # [LAW:verifiable-goals] a scan that found nothing because it could see nothing would pass the test above alike.
    assert _reads(ast.parse("import os\nos.environ.get('X')\nfrom os import getenv\n")) == [2, 3]
    assert any(_reads(ast.parse(path.read_text())) for path in (SOURCE / "daemon").rglob("*.py"))
