"""Spec rule: cudaq is imported only under src/q2jepa/quantum/, qiskit only by scripts/eval_hardware.py (+ its test)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "cudaq": lambda p: p.is_relative_to("src/q2jepa/quantum"),
    "qiskit": lambda p: p.name in {"eval_hardware.py", "test_eval_hardware.py", "test_born_equivalence.py"},
}


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_import_boundary(module):
    files = [p.relative_to(ROOT) for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py")]
    offenders = [str(p) for p in files if module in _imported_roots(ROOT / p) and not ALLOWED[module](p)]
    assert not offenders, f"{module} imported outside its boundary: {offenders}"
