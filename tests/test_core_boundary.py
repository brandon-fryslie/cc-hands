"""core decides and never acts: it imports only data modules and itself."""

import ast
from pathlib import Path

import hands.core

ALLOWED = ("collections.abc", "dataclasses", "pathlib", "typing", "hands.core")


def imported(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        match node:
            case ast.Import(names=aliases):
                names.update(alias.name for alias in aliases)
            case ast.ImportFrom(module=module, level=0) if module is not None:
                names.add(module)
            case ast.ImportFrom():
                names.add("hands.core")  # relative, so inside core
            case _:
                pass
    return names


def forbidden(source: str) -> set[str]:
    return {name for name in imported(source) if not any(name == a or name.startswith(f"{a}.") for a in ALLOWED)}


def test_core_imports_nothing_that_reaches_the_world() -> None:
    core = Path(hands.core.__file__).parent
    modules = sorted(core.rglob("*.py"))
    assert modules
    offenders = {module.name: forbidden(module.read_text()) for module in modules}
    assert {name: found for name, found in offenders.items() if found} == {}


def test_the_check_catches_what_the_architecture_names() -> None:
    source = "import subprocess\nimport socket\nfrom asyncio import streams\nfrom pipecat.frames import frames\ndef f():\n    import hands.sessions.shim\n"
    assert forbidden(source) == {"subprocess", "socket", "asyncio", "pipecat.frames", "hands.sessions.shim"}
