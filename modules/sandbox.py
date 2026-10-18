"""Restricted execution of LLM-generated pandas code.

Layers of protection:
1. AST validation: only whitelisted imports, no dangerous names/attributes.
2. Restricted builtins: no open(), eval(), exec(), etc.
3. Separate process with a timeout, so infinite loops get killed.
4. The code only ever sees a COPY of the dataframe.

NOTE: this reduces risk but is not a perfect sandbox. For production use a
container or a service built for sandboxing (Docker, E2B, etc.).
"""

import ast
import builtins
import multiprocessing as mp
import pickle
from queue import Empty

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ALLOWED_IMPORTS = {"pandas", "numpy", "plotly", "math", "datetime", "statistics"}

BLOCKED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    "exit", "quit", "help", "memoryview",
}

# File / network / system related methods
BLOCKED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_parquet",
    "read_html", "read_sql", "read_table", "read_feather", "read_hdf",
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_hdf",
    "to_feather", "write_html", "write_image", "write_json", "show",
}

SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "int", "isinstance", "len", "list", "map", "max", "min", "pow",
    "print", "range", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "Exception", "ValueError", "KeyError",
    "TypeError", "IndexError", "ZeroDivisionError",
]


def validate_code(code: str):
    """Return (is_ok, message). Rejects anything that is not plain data analysis."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"Import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                return False, f"Import not allowed: {node.module}"
        elif isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                return False, f"Use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BLOCKED_ATTRS:
                return False, f"Use of '.{node.attr}' is not allowed"
    return True, ""


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"Import not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _worker(code, df, queue):
    """Runs in a separate process."""
    try:
        safe_builtins = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES}
        safe_builtins["__import__"] = _guarded_import
        env = {
            "__builtins__": safe_builtins,
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
        }
        exec(code, env)

        if "result" not in env:
            queue.put(("error", "The code did not set a variable called `result`."))
            return

        result = env["result"]
        try:
            pickle.dumps(result)
        except Exception:
            queue.put(("error", f"Result of type {type(result).__name__} is not supported."))
            return

        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def run_code(code: str, df: pd.DataFrame, timeout: int = 20) -> dict:
    """Validate and run code. Returns {"ok": bool, "result": ..., "error": str}."""
    ok, message = validate_code(code)
    if not ok:
        return {"ok": False, "result": None, "error": message}

    ctx = mp.get_context("spawn")  # works on Windows, Mac and Linux
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, df, queue), daemon=True)
    proc.start()

    try:
        status, payload = queue.get(timeout=timeout)
    except Empty:
        proc.terminate()
        proc.join()
        return {"ok": False, "result": None, "error": f"Timed out after {timeout} seconds."}

    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()

    if status == "ok":
        return {"ok": True, "result": payload, "error": ""}
    return {"ok": False, "result": None, "error": payload}
