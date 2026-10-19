"""Static checks and restricted builtins for candidate code.

The candidate runs in a separate process (sandbox_runner.py) with only the
builtins below. Before it runs, the AST is checked: no imports, no dunder
names or attributes (blocks the usual `().__class__.__subclasses__()` escapes),
no class definitions. These are defence in depth; the real guarantee is that
the evaluator re-validates and re-simulates the returned gate list itself.
"""

import ast
import builtins

_ALLOWED_BUILTINS = [
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "frozenset", "hex", "int", "isinstance", "iter", "len",
    "list", "map", "max", "min", "next", "oct", "ord", "pow", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "IndexError", "KeyError", "AssertionError",
    "ZeroDivisionError", "StopIteration", "ArithmeticError", "RuntimeError",
]


def _quiet_print(*args, **kwargs):
    return None


def safe_builtins():
    d = {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}
    d["print"] = _quiet_print
    return d


class SourceRejected(Exception):
    pass


def check_source(src):
    """Raise SourceRejected with a model-readable message if `src` is not allowed."""
    try:
        tree = ast.parse(src, filename="<candidate>")
    except SyntaxError as e:
        raise SourceRejected(f"syntax error at line {e.lineno}: {e.msg}") from None
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SourceRejected(f"line {line}: imports are not allowed")
        if isinstance(node, ast.ClassDef):
            raise SourceRejected(f"line {line}: class definitions are not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            continue
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.arg):
            name = node.arg
        if name is not None and name.startswith("__"):
            raise SourceRejected(f"line {line}: names starting with '__' are not allowed ({name})")
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_circuit"]
    if not defs:
        raise SourceRejected("no top-level `def build_circuit(c, n, p)` found")
    return tree
