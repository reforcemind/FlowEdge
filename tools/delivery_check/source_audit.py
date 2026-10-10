"""Static inventory of control-relevant calls in the two evaluation runners.

This is a change detector for the declared instrumentation boundary.  It is not
a whole-program effect analysis: dynamic dispatch, native code, and callbacks
remain outside its scope.
"""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path


TARGETS = {
    "integrations/lerobot/src/flowedge_lerobot/rollout.py": {
        "run_rollout": Counter(
            {
                "trace.send": 2,
                "robot.send_action": 2,
                "np.copyto": 1,
                "trace.stop": 1,
            }
        )
    },
    "integrations/lerobot/src/flowedge_lerobot/async_rollout.py": {
        "run_async_rollout": Counter(
            {
                "actions.extend": 2,
                "actions.clear": 1,
                "actions.popleft": 1,
                "trace.send": 1,
                "robot.send_action": 1,
                "executor.submit": 1,
                "pending.result": 2,
                "trace.stop": 1,
            }
        )
    },
    "integrations/lerobot/src/flowedge_lerobot/delivery_trace.py": {
        "DeliveryTrace.send": Counter({"robot.send_action": 1}),
        "DeliveryTrace.stop": Counter({"robot.stop": 1}),
    },
}

CONTROL_CALLS = {
    call
    for functions in TARGETS.values()
    for expected in functions.values()
    for call in expected
}
EXTRA_CONTROL_CALLS = {"actions.append"}
WATCHED_METHODS = {"send_action", "stop", "submit", "result", "popleft"}


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


class _Inventory(ast.NodeVisitor):
    def __init__(self, source: str):
        self.source = source
        self.scope: list[str] = []
        self.guards: list[str] = []
        self.sites: list[dict] = []
        self.aliases: dict[str, str] = {}

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        value = _name(node.value)
        if value in CONTROL_CALLS or value in EXTRA_CONTROL_CALLS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = value
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        condition = ast.unparse(node.test)
        self.guards.append(condition)
        for child in node.body:
            self.visit(child)
        self.guards.pop()
        self.guards.append(f"not ({condition})")
        for child in node.orelse:
            self.visit(child)
        self.guards.pop()

    def visit_Call(self, node: ast.Call) -> None:
        syntax = _name(node.func)
        call = self.aliases.get(syntax, syntax)
        watched = call in CONTROL_CALLS or call in EXTRA_CONTROL_CALLS
        if call and call.rsplit(".", 1)[-1] in WATCHED_METHODS:
            watched = True
        if watched:
            self.sites.append(
                {
                    "scope": ".".join(self.scope),
                    "call": call,
                    "syntax": syntax,
                    "via_alias": syntax in self.aliases,
                    "line": node.lineno,
                    "guards": list(self.guards),
                }
            )
        self.generic_visit(node)


def _guard_checks(sites: list[dict]) -> list[str]:
    errors = []
    for site in sites:
        if site["scope"] not in ("run_rollout", "run_async_rollout"):
            continue
        guards = " and ".join(site["guards"])
        if site["call"] == "trace.send" and "tracing" not in guards:
            errors.append(f"line {site['line']}: trace.send is not guarded by tracing")
        if site["call"] == "robot.send_action" and "not (tracing)" not in guards:
            errors.append(
                f"line {site['line']}: direct send is not in the non-tracing branch"
            )
    return errors


def audit_sources(root: str | Path, *, overrides: dict[str, str] | None = None) -> dict:
    """Compare the live call inventory with the reviewed boundary manifest."""
    root = Path(root)
    overrides = overrides or {}
    files = []
    failures = []
    for relative, expected_scopes in TARGETS.items():
        source = overrides.get(relative)
        if source is None:
            source = (root / relative).read_text(encoding="utf-8")
        visitor = _Inventory(source)
        visitor.visit(ast.parse(source, filename=relative))
        actual_by_scope = {
            scope: Counter(
                site["call"] for site in visitor.sites if site["scope"] == scope
            )
            for scope in expected_scopes
        }
        for scope, expected in expected_scopes.items():
            if actual_by_scope[scope] != expected:
                failures.append(
                    {
                        "file": relative,
                        "scope": scope,
                        "expected": dict(expected),
                        "actual": dict(actual_by_scope[scope]),
                    }
                )
        failures.extend(
            {"file": relative, "reason": reason}
            for reason in _guard_checks(visitor.sites)
        )
        failures.extend(
            {
                "file": relative,
                "reason": f"line {site['line']}: indirect control-call alias is unsupported",
            }
            for site in visitor.sites
            if site["via_alias"]
        )
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "sites": visitor.sites,
            }
        )
    return {
        "scope": "static_source_inventory",
        "files": files,
        "failures": failures,
        "limitations": [
            "no dynamic-dispatch or callback effect analysis",
            "simple local aliases are rejected rather than resolved as equivalent source",
            "no path-feasibility proof",
            "native and library implementations are outside the inventory",
        ],
        "verdict": "pass" if not failures else "fail",
    }
