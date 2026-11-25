"""The preprocessing scripts must stay liftable: no import from the package that drives them.

``preprocess/*.py`` are invoked by the driver as subprocess CLIs, never imported, so each keeps its
own interpreter and deps. An import from ``readyneuro`` would silently work in this checkout and only
fail when the script is run anywhere else. This is a static AST check -- it catches lazy in-function
imports too, and needs none of the external tools.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "preprocess"
SRC = ROOT / "src"
PKG = SRC / "readyneuro"


def _imported_modules(path: Path) -> set[str]:
    """Every dotted module name imported by ``path`` (both ``import x`` and ``from x import y``)."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_preprocess_scripts_import_nothing_from_the_package():
    scripts = sorted(PREPROCESS.glob("*.py"))
    assert scripts, f"no preprocessing scripts under {PREPROCESS}"
    for script in scripts:
        leaked = {m for m in _imported_modules(script) if m.split(".")[0] in ("readyneuro", "brainmarks")}
        assert not leaked, f"{script.name} must import nothing from readyneuro/brainmarks, got {sorted(leaked)}"


def test_the_io_leaf_imports_nothing_from_the_package():
    """``io.py`` is the third-party-only leaf both ``tools/`` and the finalizers import DOWNWARD.
    An upward import there inverts the layer graph and makes the tools uncallable in isolation."""
    leaked = {m for m in _imported_modules(PKG / "io.py") if m.startswith("readyneuro")}
    assert not leaked, f"io.py must import nothing from readyneuro, got {sorted(leaked)}"


def test_storage_tools_do_not_import_the_builder():
    for name in ("codec", "value_preproc", "shards", "eval_arrow"):
        leaked = {m for m in _imported_modules(PKG / "tools" / f"{name}.py") if m.startswith("readyneuro.builder")}
        assert not leaked, f"tools/{name}.py must not import builder/, got {sorted(leaked)}"


def test_every_import_in_the_package_is_stdlib_or_a_declared_dependency():
    """Catches the two ways a release ships something a fresh checkout cannot import: a leftover
    reference to a module that was renamed during extraction, and a dependency that is used but not
    declared. Both work perfectly in the environment they were written in. Covers BOTH wheel
    packages -- ``src/readyneuro`` and the ``src/brainmarks`` overlay."""
    import re
    import sys
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    requirements = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    # distribution name -> import name, where they differ
    aliases = {"pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn"}
    declared = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].strip().lower()
        declared.add(aliases.get(name, name.replace("-", "_")))
    allowed = declared | set(sys.stdlib_module_names) | {"readyneuro", "brainmarks"}

    for module in sorted(SRC.rglob("*.py")):
        roots = {m.split(".")[0] for m in _imported_modules(module)}
        undeclared = {r for r in roots if r not in allowed and not r.startswith("_")}
        assert not undeclared, f"{module.relative_to(ROOT)} imports undeclared {sorted(undeclared)}"
