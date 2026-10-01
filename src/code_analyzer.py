"""Static checks and constrained runtime diagnostics for Python source."""
import ast
import builtins
import contextlib
import ctypes
import importlib.machinery
import io
import json
import math
import operator
import os
import re
import subprocess
import sys
import traceback


def _positive_int_setting(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, 1), maximum)


def _positive_float_setting(name: str, default: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, 0.1), maximum)


MAX_SOURCE_LENGTH = _positive_int_setting("CODEERRORBOT_MAX_SOURCE_CHARS", 100000, 2000000)
MAX_AST_NODES = _positive_int_setting("CODEERRORBOT_MAX_AST_NODES", 20000, 200000)
MAX_RANGE_SIZE = 100
EXECUTION_TIMEOUT_SECONDS = _positive_float_setting("CODEERRORBOT_EXECUTION_TIMEOUT", 2.0, 30.0)
MAX_CHILD_MEMORY_BYTES = _positive_int_setting("CODEERRORBOT_MAX_MEMORY_MB", 128, 1024) * 1024 * 1024
_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
    "input", "max", "min", "print", "range", "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
}
_SAFE_METHOD_NAMES = {"append", "get", "keys", "values", "items"}
_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.FunctionDef, ast.arguments,
    ast.arg, ast.Return, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.If, ast.For, ast.While,
    ast.Break, ast.Continue, ast.Pass, ast.Call, ast.Attribute, ast.keyword, ast.Assert,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub,
    ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Load, ast.Store,
}
_PIP_PACKAGE_NAMES = {
    "cv2": "opencv-python", "sklearn": "scikit-learn", "PIL": "Pillow",
    "yaml": "PyYAML", "bs4": "beautifulsoup4", "Crypto": "pycryptodome",
    "dotenv": "python-dotenv", "Bio": "biopython", "gi": "PyGObject",
}


def _bounded_range(*args):
    values = range(*args)
    if len(values) > MAX_RANGE_SIZE:
        raise ValueError(f"range is limited to {MAX_RANGE_SIZE} iterations during analysis")
    return values


def _safe_print(*values, **kwargs):
    if set(kwargs) - {"sep", "end"}:
        raise TypeError("print options are restricted during analysis")
    separator = kwargs.get("sep", " ")
    ending = kwargs.get("end", "\n")
    if not isinstance(separator, str) or not isinstance(ending, str) or len(separator) > 100 or len(ending) > 100:
        raise ValueError("print separator and ending must be short strings during analysis")
    text = separator.join(str(value) for value in values)
    if len(text) > 1000:
        text = text[:1000] + "... [output truncated]"
    sys.stdout.write(text + ending)


def _bounded_input():
    def reject_input(prompt=""):
        raise RuntimeError("input() requires explicit input, which is unavailable during analysis")
    return reject_input


def _safe_add(left, right):
    if type(left) in (str, list, tuple) and type(left) is type(right):
        if len(left) + len(right) > 10000:
            raise ValueError("Combined sequence is too large for constrained analysis")
    result = operator.add(left, right)
    if type(result) is int and result.bit_length() > 8192:
        raise ValueError("Integer result is too large for constrained analysis")
    return result


def _safe_multiply(left, right):
    sequence, count = (left, right) if type(left) in (str, list, tuple) else (right, left)
    if type(sequence) in (str, list, tuple) and type(count) is int:
        if len(sequence) * max(0, count) > 10000:
            raise ValueError("Repeated sequence is too large for constrained analysis")
    result = operator.mul(left, right)
    if type(result) is int and result.bit_length() > 8192:
        raise ValueError("Integer result is too large for constrained analysis")
    return result


def _safe_append(sequence, value):
    if type(sequence) is not list:
        raise AttributeError(f"'{type(sequence).__name__}' object has no attribute 'append'")
    if len(sequence) >= MAX_RANGE_SIZE * 100:
        raise ValueError("List growth is limited during constrained analysis")
    sequence.append(value)


class _BoundedArithmetic(ast.NodeTransformer):
    _helpers = {ast.Add: "__safe_add", ast.Mult: "__safe_multiply"}

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        helper = self._helpers.get(type(node.op))
        if helper is None:
            return node
        return ast.copy_location(
            ast.Call(func=ast.Name(id=helper, ctx=ast.Load()), args=[node.left, node.right], keywords=[]),
            node,
        )

    def visit_AugAssign(self, node):
        node = self.generic_visit(node)
        helper = self._helpers.get(type(node.op))
        if helper is None:
            return node
        target_load = ast.Name(id=node.target.id, ctx=ast.Load())
        call = ast.Call(func=ast.Name(id=helper, ctx=ast.Load()), args=[target_load, node.value], keywords=[])
        return ast.copy_location(ast.Assign(targets=[node.target], value=call), node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "append":
            return ast.copy_location(
                ast.Call(func=ast.Name(id="__safe_append", ctx=ast.Load()), args=[node.func.value, *node.args], keywords=[]),
                node,
            )
        return node


def _is_bounded_while(tree: ast.AST, loop: ast.While) -> bool:
    test = loop.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    if not isinstance(test.left, ast.Name) or not isinstance(test.comparators[0], ast.Constant):
        return False
    if type(test.comparators[0].value) is not int:
        return False
    counter = test.left.id
    final_update = loop.body[-1] if loop.body else None
    if not isinstance(final_update, ast.AugAssign) or not isinstance(final_update.target, ast.Name) or final_update.target.id != counter:
        return False
    if not isinstance(final_update.value, ast.Constant) or type(final_update.value.value) is not int or final_update.value.value <= 0:
        return False
    op = type(test.ops[0])
    update_op = type(final_update.op)
    increasing = op in (ast.Lt, ast.LtE) and update_op is ast.Add
    decreasing = op in (ast.Gt, ast.GtE) and update_op is ast.Sub
    if not (increasing or decreasing):
        return False
    if any(isinstance(node, (ast.Break, ast.Continue)) for node in ast.walk(ast.Module(body=loop.body, type_ignores=[]))):
        return False

    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list) or loop not in body:
            continue
        loop_index = body.index(loop)
        start = None
        for statement in body[:loop_index]:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id == counter and isinstance(statement.value, ast.Constant) and type(statement.value.value) is int:
                start = statement.value.value
        if start is None:
            return False
        distance = test.comparators[0].value - start if increasing else start - test.comparators[0].value
        step = final_update.value.value
        extra = 1 if isinstance(test.ops[0], (ast.LtE, ast.GtE)) else 0
        return max(0, (distance + step - 1) // step + extra) <= MAX_RANGE_SIZE
    return False


def _validate_ast(tree: ast.AST) -> str | None:
    nodes = list(ast.walk(tree))
    local_functions = {
        node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if len(nodes) > MAX_AST_NODES:
        return "This program is too complex for the constrained analyzer."

    for node in nodes:
        if type(node) not in _ALLOWED_NODES:
            return f"{type(node).__name__} is not supported in constrained analysis."
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "Names beginning with double underscores are not allowed."
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _SAFE_METHOD_NAMES:
                return f"Attribute access to '{node.attr}' is not allowed during analysis."
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_BUILTIN_NAMES and node.func.id not in local_functions:
                    return f"Calling '{node.func.id}' is not allowed during analysis."
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _SAFE_METHOD_NAMES:
                    return f"Calling '{node.func.attr}' is not allowed during analysis."
            else:
                return "This function call is not supported in constrained analysis."
            if isinstance(node.func, ast.Attribute) and node.func.attr == "extend":
                return "List extension is not supported during constrained analysis."
        if isinstance(node, ast.While) and not _is_bounded_while(tree, node):
            return "This while loop has no statically verifiable iteration bound and was not executed."
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return "Exponentiation is not executed because it can exhaust resources."
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name) or isinstance(node.op, ast.Pow):
                return "This augmented assignment is not supported during constrained analysis."

    return None


def _blocked_operation_result(tree: ast.AST, source: str) -> dict | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            return {"status": "unsupported", "message": "File access is not inspected or performed during constrained analysis.", "source": source, "line": node.lineno}
    return None


def _inspect_imports(tree: ast.AST) -> list[dict]:
    imports = []
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_names = [node.module] if node.module else []
            if node.level:
                module_names = [("." * node.level) + (node.module or "")]
        else:
            continue

        for module_name in module_names:
            if module_name in seen:
                continue
            seen.add(module_name)
            if module_name.startswith("."):
                imports.append({
                    "module": module_name,
                    "line": node.lineno,
                    "kind": "relative project import",
                    "status": "not checked",
                })
                continue
            top_level = module_name.split(".", 1)[0]
            item = {"module": module_name, "line": node.lineno}
            if top_level in sys.stdlib_module_names or top_level in sys.builtin_module_names:
                spec = _find_module_spec(module_name)
                status = "available" if top_level in sys.builtin_module_names or spec is not None else "not found"
                item.update({"kind": "standard library", "status": status})
            else:
                spec = _find_module_spec(module_name)
                if spec is None:
                    package_name = _PIP_PACKAGE_NAMES.get(top_level, top_level)
                    item.update({
                        "kind": "third-party or local module",
                        "status": "not found",
                        "package": package_name,
                        "install_command": f"python -m pip install {package_name}",
                    })
                else:
                    item.update({"kind": "third-party or local module", "status": "available"})
            imports.append(item)
    return sorted(imports, key=lambda item: (item["line"], item["module"]))


def _find_module_spec(module_name: str):
    try:
        if module_name in sys.modules:
            return sys.modules[module_name]
        parts = module_name.split(".")
        qualified_name = parts[0]
        spec = importlib.machinery.PathFinder.find_spec(qualified_name)
        for part in parts[1:]:
            if spec is None or spec.submodule_search_locations is None:
                return None
            qualified_name += f".{part}"
            if qualified_name in sys.modules:
                spec = sys.modules[qualified_name]
            else:
                spec = importlib.machinery.PathFinder.find_spec(
                    qualified_name,
                    spec.submodule_search_locations,
                )
        return spec
    except Exception:
        return None


def _location_for_exception(error: BaseException) -> tuple[int | None, int | None]:
    if isinstance(error, SyntaxError):
        return error.lineno, error.offset
    extracted = traceback.extract_tb(error.__traceback__)
    for frame in reversed(extracted):
        if frame.filename == "<submitted-code>":
            return frame.lineno, None
    return None, None


def _diagnostic(error_type: str, message: str, source: str, line: int | None,
                column: int | None = None, *, traceback_text: str = "",
                collections: dict | None = None) -> dict:
    lines = source.splitlines()
    problematic_line = lines[line - 1] if line and 0 < line <= len(lines) else ""
    result = {
        "status": "error",
        "error_type": error_type,
        "error_message": f"{error_type}: {message}",
        "line": line,
        "column": column,
        "problematic_code": problematic_line,
        "source": source,
        "traceback": traceback_text,
        "collections": collections or {},
    }
    if error_type == "NameError":
        result["additional_issues"] = _find_additional_undefined_names(source, message, line)
    return result


def _logic_warnings(tree: ast.AST, source: str) -> list[dict]:
    lines = source.splitlines()
    warnings = []
    seen = set()

    def add_warning(category, node, message, explanation):
        line = getattr(node, "lineno", None)
        column = getattr(node, "col_offset", 0) + 1
        key = (category, line, message)
        if key in seen:
            return
        seen.add(key)
        warnings.append({
            "category": category,
            "line": line,
            "column": column,
            "code": lines[line - 1] if line and line <= len(lines) else "",
            "message": message,
            "explanation": explanation,
        })

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)) and isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            value = node.test.value
            add_warning(
                "SuspiciousCondition",
                node.test,
                f"This condition is always {str(value).lower()}.",
                "A constant condition can make a branch behave differently than intended; verify the condition.",
            )
            unreachable = node.body if not value else getattr(node, "orelse", [])
            for statement in unreachable:
                add_warning(
                    "UnreachableCode",
                    statement,
                    "This statement is in a branch that the constant condition cannot select.",
                    "This code is unreachable for the condition as written; check whether the condition was intended to be constant.",
                )

        if isinstance(node, ast.Compare):
            left = node.left
            for operation, right in zip(node.ops, node.comparators):
                if isinstance(left, ast.Name) and isinstance(right, ast.Name) and left.id == right.id and isinstance(operation, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                    add_warning(
                        "SuspiciousCondition",
                        node,
                        f"This condition compares `{left.id}` with itself.",
                        "Comparing a value with itself is often accidental and may make a condition always true or false.",
                    )
                left = right

        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            exited = False
            for statement in statements:
                if exited:
                    add_warning(
                        "UnreachableCode",
                        statement,
                        "This statement follows an unconditional return, break, or continue in the same block.",
                        "Python will leave this block before reaching this statement.",
                    )
                elif isinstance(statement, (ast.Return, ast.Break, ast.Continue)):
                    exited = True

    return warnings


def _static_checks(source: str, tree: ast.AST) -> tuple[list[dict], str | None]:
    logic_warnings = _logic_warnings(tree, source)
    try:
        from pyflakes.api import check
    except ImportError:
        return logic_warnings, "Pyflakes is not installed. Install it with `python -m pip install pyflakes`."

    class Reporter:
        def __init__(self):
            self.issues = []

        def flake(self, message):
            lines = source.splitlines()
            line = message.lineno
            explanation = _explain_static_message(message)
            self.issues.append({
                "category": type(message).__name__,
                "line": line,
                "column": message.col + 1,
                "code": lines[line - 1] if line and line <= len(lines) else "",
                "message": message.message,
                "explanation": explanation,
            })

        def syntaxError(self, filename, message, lineno, offset, text):
            return None

        def unexpectedError(self, filename, message):
            return None

    reporter = Reporter()
    try:
        check(source, "<submitted-code>", reporter)
    except Exception:
        return logic_warnings + reporter.issues, "Pyflakes could not analyze this source. Reinstall it with `python -m pip install --upgrade pyflakes`."
    return logic_warnings + reporter.issues, None


def _explain_static_message(message) -> str:
    category = type(message).__name__
    if category == "UndefinedName":
        return "This name is used but is not defined. Check its spelling or define it before this line."
    if category == "UnusedImport":
        return "This module or name is imported but never used. Remove the import if it is not needed."
    if category == "UnusedVariable":
        return "This variable is assigned but never used. Check whether a later statement is missing."
    if category == "RedefinedWhileUnused":
        return "This name replaces an earlier import that was never used. Check for a naming conflict."
    if category == "ImportShadowedByLoopVar":
        return "A loop variable hides an imported name. Rename the loop variable to avoid confusion."
    return "This is a possible source-code issue reported by static analysis, not a confirmed runtime error."


def _find_additional_undefined_names(source: str, message: str, line: int | None) -> list[dict]:
    match = re.search(r"name '([^']+)' is not defined", message)
    if not match:
        return []
    first_name = match.group(1)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    defined = _SAFE_BUILTIN_NAMES | {"True", "False", "None"}
    loaded = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load) and node.id not in defined:
                loaded.append(node)
    issues = []
    seen = {first_name}
    for node in sorted(loaded, key=lambda item: (item.lineno, item.col_offset)):
        if node.id in seen or (line is not None and node.lineno <= line):
            continue
        seen.add(node.id)
        issues.append({
            "error_type": "NameError",
            "line": node.lineno,
            "problematic_code": source.splitlines()[node.lineno - 1],
            "message": f"'{node.id}' is also read but is not assigned in the program.",
        })
    return issues


class _SkipFailingLines(ast.NodeTransformer):
    def __init__(self, lines: set[int]):
        self.lines = lines

    def generic_visit(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                updated = []
                for item in value:
                    if isinstance(item, ast.stmt) and getattr(item, "lineno", None) in self.lines:
                        replacement = ast.Pass()
                        ast.copy_location(replacement, item)
                        updated.append(replacement)
                    else:
                        updated.append(self.visit(item) if isinstance(item, ast.AST) else item)
                setattr(node, field, updated)
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node


def analyze_source(source: str, timeout: float = EXECUTION_TIMEOUT_SECONDS) -> dict:
    """Analyze source statically, then optionally diagnose it in a bounded child."""
    if not source or not source.strip():
        return {"status": "empty", "message": "Please enter Python source code."}
    if len(source) > MAX_SOURCE_LENGTH:
        return {
            "status": "unsupported",
            "message": f"This submission has {len(source):,} characters; the configured limit is {MAX_SOURCE_LENGTH:,}. Increase CODEERRORBOT_MAX_SOURCE_CHARS to change it.",
        }

    try:
        tree = ast.parse(source)
        compile(tree, "<submitted-code>", "exec")
    except (SyntaxError, IndentationError) as error:
        line, column = _location_for_exception(error)
        return _diagnostic(type(error).__name__, error.msg, source, line, column)

    import_checks = _inspect_imports(tree)
    missing_imports = [
        item for item in import_checks
        if item["status"] == "not found" and item["kind"] != "standard library"
    ]
    static_warnings, dependency_notice = _static_checks(source, tree)

    def include_static_results(result):
        result["warnings"] = static_warnings
        result["import_checks"] = import_checks
        if dependency_notice:
            result["dependency_notice"] = dependency_notice
        return result

    if missing_imports:
        return include_static_results({
            "status": "missing_dependencies",
            "message": "One or more imports were not found in the active Python environment. No package was installed, and submitted code was not imported or executed.",
            "source": source,
            "missing_dependencies": missing_imports,
        })

    blocked_result = _blocked_operation_result(tree, source)
    if blocked_result:
        return include_static_results(blocked_result)

    unsupported = _validate_ast(tree)
    if unsupported:
        return include_static_results({
            "status": "unsupported",
            "message": unsupported,
            "source": source,
            "line_count": len(source.splitlines()),
            "functions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
            "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
        })

    errors = []
    skipped_lines = set()
    for _ in range(12):
        try:
            completed = _run_bounded_child(source, skipped_lines, timeout)
        except subprocess.TimeoutExpired:
            if errors:
                break
            return include_static_results({"status": "timeout", "message": f"Execution stopped after {timeout:g} seconds. Static analysis completed."})
        except (OSError, RuntimeError) as error:
            return include_static_results({
                "status": "execution_unavailable",
                "message": f"Static analysis completed, but bounded runtime diagnostics are unavailable: {error}",
                "source": source,
            })

        if completed.returncode and not completed.stdout.strip():
            return include_static_results({
                "status": "resource_limit",
                "message": "The constrained child process was stopped by an operating-system resource limit. Static analysis completed; runtime diagnostics are unavailable for this submission.",
                "source": source,
            })

        try:
            child_result = json.loads(completed.stdout) if completed.stdout.strip() else {}
        except json.JSONDecodeError:
            return {"status": "execution_error", "message": "The constrained analysis process returned an invalid result."}
        if child_result.get("ok"):
            break

        error_type = child_result.get("type", "RuntimeError")
        message = child_result.get("message", "The submitted code could not be analyzed.")
        line = child_result.get("line")
        issue = _diagnostic(
            error_type,
            message,
            source,
            line,
            traceback_text=child_result.get("traceback", ""),
            collections=child_result.get("collections", {}),
        )
        if not line or line in skipped_lines or any(item["line"] == line and item["error_type"] == error_type for item in errors):
            break
        errors.append(issue)
        skipped_lines.add(line)

    if not errors:
        assigned = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)})
        return include_static_results({
            "status": "ok",
            "message": "The code passed syntax validation and no runtime error was found in the constrained analysis run.",
            "line_count": len(source.splitlines()),
            "functions": sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)),
            "variables": assigned,
            "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
        })

    first_error = errors[0]
    first_error["errors"] = [dict(issue) for issue in errors]
    first_error["additional_issues"] = [
        {"error_type": issue["error_type"], "line": issue["line"],
         "problematic_code": issue["problematic_code"], "message": issue["error_message"]}
        for issue in errors[1:]
    ]
    first_error["warnings"] = static_warnings
    if dependency_notice:
        first_error["dependency_notice"] = dependency_notice
    first_error["analysis_note"] = (
        "Additional independent failures were sought by rerunning the constrained program with each failing statement skipped. "
        "Dependent runtime failures may still require another analysis after correction."
    )
    return first_error


def _run_bounded_child(source: str, skipped_lines: set[int], timeout: float) -> subprocess.CompletedProcess:
    args = [sys.executable, "-I", os.path.abspath(__file__), "--run", json.dumps(sorted(skipped_lines))]
    environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}

    if os.name == "nt":
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise RuntimeError("Windows could not create a resource-limited job object")

        process = None
        try:
            limits = ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = 0x8 | 0x100 | 0x200 | 0x2000
            limits.BasicLimitInformation.ActiveProcessLimit = 1
            limits.ProcessMemoryLimit = MAX_CHILD_MEMORY_BYTES
            limits.JobMemoryLimit = MAX_CHILD_MEMORY_BYTES
            if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise RuntimeError("Windows could not configure child resource limits")

            process = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=os.path.dirname(os.path.abspath(__file__)), env=environment,
            )
            if not kernel32.AssignProcessToJobObject(job, process._handle):
                raise RuntimeError("Windows could not assign the child to its resource-limited job")
            try:
                stdout, stderr = process.communicate(source, timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            kernel32.CloseHandle(job)

    try:
        import resource
    except ImportError as error:
        raise RuntimeError("this platform does not expose process resource limits") from error

    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (MAX_CHILD_MEMORY_BYTES, MAX_CHILD_MEMORY_BYTES))
        cpu_seconds = max(1, math.ceil(timeout))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

    return subprocess.run(
        args, input=source, capture_output=True, text=True, timeout=timeout,
        cwd=os.path.dirname(os.path.abspath(__file__)), env=environment,
        preexec_fn=set_limits,
    )


def _safe_snapshot(value, depth: int = 0):
    if depth > 3:
        return "..."
    if value is None or type(value) in (str, int, float, bool):
        return value if not isinstance(value, str) or len(value) <= 120 else value[:117] + "..."
    if type(value) in (list, tuple):
        return [_safe_snapshot(item, depth + 1) for item in value[:10]]
    if type(value) is dict:
        return {str(key)[:80]: _safe_snapshot(item, depth + 1) for key, item in list(value.items())[:10] if type(key) in (str, int, float, bool)}
    return None


def _run_child(source: str, skipped_lines: set[int] | None = None) -> None:
    output_capture = io.StringIO()
    try:
        tree = ast.parse(source)
        reason = _validate_ast(tree)
        if reason:
            raise RuntimeError(reason)
        safe_builtins = {
            name: getattr(builtins, name)
            for name in _SAFE_BUILTIN_NAMES
            if name not in {"range", "print", "input"}
        }
        safe_builtins["range"] = _bounded_range
        safe_builtins["print"] = _safe_print
        safe_builtins["input"] = _bounded_input()
        safe_builtins["__safe_append"] = _safe_append
        namespace = {"__builtins__": safe_builtins}
        if skipped_lines:
            tree = _SkipFailingLines(skipped_lines).visit(tree)
            ast.fix_missing_locations(tree)
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(io.StringIO()):
            safe_builtins["__safe_add"] = _safe_add
            safe_builtins["__safe_multiply"] = _safe_multiply
            bounded_tree = _BoundedArithmetic().visit(tree)
            ast.fix_missing_locations(bounded_tree)
            exec(compile(bounded_tree, "<submitted-code>", "exec"), namespace, namespace)
        print(json.dumps({"ok": True, "output": output_capture.getvalue()}))
    except BaseException as error:
        line, _ = _location_for_exception(error)
        collections = {
            name: {"type": "list", "length": len(value), "sample": _safe_snapshot(value)}
            for name, value in locals().get("namespace", {}).items()
            if not name.startswith("_") and type(value) is list
        }
        print(json.dumps({
            "ok": False,
            "type": type(error).__name__,
            "message": str(error),
            "line": line,
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "collections": collections,
            "output": output_capture.getvalue(),
        }))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--run":
    _run_child(sys.stdin.read(), set(json.loads(sys.argv[2])))