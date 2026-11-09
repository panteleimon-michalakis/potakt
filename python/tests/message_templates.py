"""Every validation message template in the model sources and in ``source_issues`` (TASK-005e1).

A template is the constant text of a message with ``{}`` for each replacement field. It is
collected from the abstract syntax tree when the message reaches a *sink*, i.e. the message
argument of ``raise_errors`` (a list of ``(location, message)`` pairs), ``custom_error``,
``issue_at``, ``ValueError``, ``<reader>.report``, ``<reader>.cell_error``, ``ValidationIssue`` or
``PydanticCustomError``, in one of these ways:

- (a) as the argument itself (a string or an f-string);
- (b) through a local name assigned (``=`` or ``:=``) in the same function;
- (c) through a module-level constant (also a value of a module-level dict);
- (d) as the return value of a function of the scanned modules called there; a function whose
  result is a replacement field of a collected f-string contributes its own returns too
  (``limit {above}``).

An argument that only passes a message on (a parameter, ``str(error)``, ``"; ".join(...)``) is the
template ``{}``.
"""

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HOLE = "{}"
# Sink name -> position of its message argument.
SINKS = {
    "custom_error": 0,
    "ValueError": 0,
    "issue_at": 1,
    "report": 1,
    "cell_error": 4,
    "raise_errors": 1,
    "ValidationIssue": 1,
    "PydanticCustomError": 1,
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _assigned(node: ast.AST, name: str) -> ast.expr | None:
    """The value ``node`` assigns to ``name`` (``=``, annotated ``=`` or ``:=``), if any."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.NamedExpr):
        targets = [node.target]
    else:
        return None
    named = any(isinstance(target, ast.Name) and target.id == name for target in targets)
    return node.value if named else None


@dataclass
class _Module:
    tree: ast.Module
    functions: dict[str, ast.FunctionDef] = field(default_factory=dict[str, ast.FunctionDef])
    constants: dict[str, ast.expr] = field(default_factory=dict[str, ast.expr])


@dataclass(frozen=True)
class _Scope:
    module: _Module
    function: ast.FunctionDef | None


Reader = Callable[[ast.expr, _Scope, frozenset[str]], list[str]]


class _Collector:
    def __init__(self, sources: list[Path]) -> None:
        self.modules: list[_Module] = []
        self.parents: dict[ast.AST, ast.AST] = {}
        self.found: set[str] = set()
        for source in sources:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=source.name)
            module = _Module(tree)
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    self.parents[child] = node
                if isinstance(node, ast.FunctionDef):
                    module.functions.setdefault(node.name, node)
            for statement in tree.body:
                if isinstance(statement, ast.Assign):
                    targets = statement.targets
                elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                    targets = [statement.target]
                else:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name) and statement.value is not None:
                        module.constants[target.id] = statement.value
            self.modules.append(module)

    # --- lookups ----------------------------------------------------------------------------

    def _function(self, name: str, module: _Module) -> tuple[_Module, ast.FunctionDef] | None:
        """``name`` in its own module first, then in another scanned module (an import)."""
        for candidate in (module, *self.modules):
            if name in candidate.functions:
                return candidate, candidate.functions[name]
        return None

    def _constant(self, name: str, module: _Module) -> tuple[_Module, ast.expr] | None:
        for candidate in (module, *self.modules):
            if name in candidate.constants:
                return candidate, candidate.constants[name]
        return None

    def _enclosing(self, node: ast.AST) -> ast.FunctionDef | None:
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, ast.FunctionDef):
                return node
        return None

    def _locals(self, name: str, scope: _Scope) -> list[ast.expr]:
        if scope.function is None:
            return []
        values = [_assigned(node, name) for node in ast.walk(scope.function)]
        return [value for value in values if value is not None]

    # --- messages ---------------------------------------------------------------------------

    def message(self, expr: ast.expr, scope: _Scope, seen: frozenset[str]) -> list[str]:
        """The templates a message expression can produce."""
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return [expr.value]
        if isinstance(expr, ast.JoinedStr):
            text = ""
            for value in expr.values:
                if isinstance(value, ast.FormattedValue):
                    text += HOLE
                    self._nested(value.value, scope, seen)
                elif isinstance(value, ast.Constant):
                    text += str(value.value)
            return [text]
        if isinstance(expr, ast.IfExp):
            return [*self.message(expr.body, scope, seen), *self.message(expr.orelse, scope, seen)]
        if isinstance(expr, ast.Name):
            values = self._locals(expr.id, scope)
            if values:
                return [t for value in values for t in self.message(value, scope, seen)]
            constant = self._constant(expr.id, scope.module)
            if constant is not None:
                module, value = constant
                return self.message(value, _Scope(module, None), seen)
            return [HOLE]  # a parameter: the message is passed on
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
            module, mapping = self._constant(expr.value.id, scope.module) or (scope.module, None)
            if isinstance(mapping, ast.Dict):
                return [
                    t
                    for value in mapping.values
                    for t in self.message(value, _Scope(module, None), seen)
                ]
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            return self._returns(expr.func.id, scope.module, seen, self.message)
        return [HOLE]  # str(error), "; ".join(messages), detail["msg"], ...

    def _nested(self, expr: ast.expr, scope: _Scope, seen: frozenset[str]) -> None:
        """A replacement field whose value is the result of a scanned function: its returns are
        message templates too (rule d)."""
        calls: list[ast.expr] = [expr] if isinstance(expr, ast.Call) else []
        if isinstance(expr, ast.Name):
            calls = [value for value in self._locals(expr.id, scope) if isinstance(value, ast.Call)]
        for call in calls:
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                for template in self._returns(call.func.id, scope.module, seen, self.message):
                    if re.search("[A-Za-z]", template.replace(HOLE, "")):
                        self.found.add(template)

    def errors(self, expr: ast.expr, scope: _Scope, seen: frozenset[str]) -> list[str]:
        """The message templates of a list of ``(location, message)`` pairs."""
        if isinstance(expr, ast.List):
            return [
                t
                for element in expr.elts
                if isinstance(element, ast.Tuple) and len(element.elts) == 2
                for t in self.message(element.elts[1], scope, seen)
            ]
        if isinstance(expr, ast.ListComp):
            pair = expr.elt
            if isinstance(pair, ast.Tuple) and len(pair.elts) == 2:
                return self.message(pair.elts[1], scope, seen)
            return []
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            return self._returns(expr.func.id, scope.module, seen, self.errors)
        if isinstance(expr, ast.Name) and scope.function is not None:
            found: list[str] = []
            for node in ast.walk(scope.function):
                value = _assigned(node, expr.id)
                if value is not None:
                    found += self.errors(value, scope, seen)
                elif isinstance(node, ast.AugAssign):
                    if isinstance(node.target, ast.Name) and node.target.id == expr.id:
                        found += self.errors(node.value, scope, seen)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "append"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == expr.id
                ):
                    found += self.errors(ast.List(elts=list(node.args)), scope, seen)
            return found
        return []

    def _returns(self, name: str, module: _Module, seen: frozenset[str], read: Reader) -> list[str]:
        """What ``read`` finds in the return values of function ``name``."""
        resolved = self._function(name, module)
        if resolved is None or name in seen:
            return [HOLE]
        owner, function = resolved
        scope = _Scope(owner, function)
        found: list[str] = []
        for node in ast.walk(function):
            if isinstance(node, ast.Return) and node.value is not None:
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    continue
                found += read(node.value, scope, seen | {name})
        return found

    def collect(self) -> set[str]:
        for module in self.modules:
            for node in ast.walk(module.tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node)
                if name not in SINKS or len(node.args) <= SINKS[name]:
                    continue
                argument = node.args[SINKS[name]]
                scope = _Scope(module, self._enclosing(node))
                read = self.errors if name == "raise_errors" else self.message
                self.found.update(read(argument, scope, frozenset()))
        return self.found


def collect_templates(sources: list[Path]) -> set[str]:
    """Every message template that reaches a sink in ``sources``."""
    return _Collector(sources).collect()


def template_pattern(template: str) -> re.Pattern[str]:
    """A regex that matches the messages of ``template`` (each ``{}`` matches any text)."""
    return re.compile("(.*)".join(re.escape(part) for part in template.split(HOLE)), re.DOTALL)
