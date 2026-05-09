"""
ascent/research/factor_discovery/leakage_scanner.py

Detects forward-looking data access patterns in factor code strings.
Pre-validation gate before any code runs.

Returns (is_clean: bool, message: str).
"""
from __future__ import annotations

import ast
import re
from typing import Tuple


_LOOKAHEAD_PATTERNS = [
    (r"datetime\.now\(\)",        "datetime.now() is future knowledge — use df.index[-1]"),
    (r"datetime\.today\(\)",      "datetime.today() is future knowledge"),
    (r"pd\.Timestamp\.today\(\)", "pd.Timestamp.today() is future knowledge"),
    (r"pd\.Timestamp\.now\(\)",   "pd.Timestamp.now() is future knowledge"),
    (r"time\.time\(\)",           "time.time() is future knowledge"),
    (r"\.shift\(-\d+\)",          ".shift(-N) accesses future rows — use positive shift only"),
    (r"\.tail\s*\(\s*1\s*\)",     ".tail(1) may be a lookahead pattern — use .iloc[-1] explicitly"),
]


class _LeakageVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: list = []

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "rolling"):
            for arg in node.args:
                if isinstance(arg, ast.Lambda):
                    src = ast.unparse(arg)
                    if "[-" in src:
                        self.violations.append(
                            "Negative index inside rolling().apply() lambda — potential lookahead"
                        )
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            val = node.slice.value
            if len(val) == 10 and val.count("-") == 2:
                try:
                    from datetime import date
                    parsed = date.fromisoformat(val)
                    if parsed.year > 2026:
                        self.violations.append(f"Hard-coded future date: '{val}'")
                except ValueError:
                    pass
        self.generic_visit(node)


def scan_for_leakage(code: str) -> Tuple[bool, str]:
    """
    Scan factor code string for lookahead / forward-data patterns.
    Returns (is_clean, message).
    """
    for pattern, message in _LOOKAHEAD_PATTERNS:
        if re.search(pattern, code):
            return False, f"Lookahead pattern detected: {message}"

    try:
        tree = ast.parse(code)
        visitor = _LeakageVisitor()
        visitor.visit(tree)
        if visitor.violations:
            return False, f"Structural lookahead: {'; '.join(visitor.violations)}"
    except SyntaxError:
        pass

    return True, "OK"
