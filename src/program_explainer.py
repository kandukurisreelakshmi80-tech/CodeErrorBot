"""Build beginner-friendly program explanations from Python ASTs only."""
import ast
import operator
import sys

from code_analyzer import MAX_AST_NODES, MAX_SOURCE_LENGTH


_LIBRARY_PURPOSES = {
    "collections": "provides specialized container types such as counters and deques",
    "csv": "reads and writes comma-separated tabular data",
    "datetime": "works with dates and times",
    "json": "converts data to and from JSON text",
    "math": "provides common mathematical functions and constants",
    "matplotlib": "creates plots and charts",
    "numpy": "provides arrays and efficient numerical operations",
    "os": "provides operating-system and path utilities",
    "pandas": "provides labeled tables and tools for data analysis",
    "pathlib": "provides object-based file and directory paths",
    "random": "generates pseudo-random values and selections",
    "re": "searches and transforms text using regular expressions",
    "requests": "sends HTTP requests and works with web responses",
    "seaborn": "creates statistical data visualizations",
    "sklearn": "provides machine-learning tools",
    "statistics": "provides common statistical calculations",
    "sys": "provides access to Python runtime and interpreter information",
    "tensorflow": "provides tools for building and running machine-learning models",
    "torch": "provides tensor operations and machine-learning tools",
}

_BINARY_NAMES = {
    ast.Add: "addition", ast.Sub: "subtraction", ast.Mult: "multiplication",
    ast.Div: "division", ast.FloorDiv: "floor division", ast.Mod: "remainder",
    ast.Pow: "exponentiation", ast.MatMult: "matrix multiplication",
}
_UNKNOWN = object()


def _source(node, source):
    try:
        text = ast.get_source_segment(source, node) or ast.unparse(node)
    except (AttributeError, ValueError, RecursionError):
        text = type(node).__name__
    text = " ".join(text.split())
    return text if len(text) <= 120 else text[:117] + "..."


def _line(node):
    return getattr(node, "lineno", 1)


def _names_in_target(target):
    return sorted({node.id for node in ast.walk(target) if isinstance(node, ast.Name)})


class _Facts(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.assignments = []
        self.variables = set()
        self.parameters = set()
        self.functions = []
        self.classes = []
        self.loops = []
        self.comprehensions = []
        self.conditions = []
        self.tries = []
        self.withs = []
        self.calls = []
        self.print_calls = []
        self.input_calls = []
        self.returns = []
        self.binops = []
        self.comparisons = []
        self.collections = set()
        self.loaded_names = {}
        self.attribute_uses = {}

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "local": alias.asname or alias.name.split(".")[0],
                "line": node.lineno,
            })

    def visit_ImportFrom(self, node):
        module = ("." * node.level) + (node.module or "")
        for alias in node.names:
            if alias.name == "*":
                self.imports.append({"module": module, "local": "*", "line": node.lineno})
            else:
                self.imports.append({
                    "module": module,
                    "local": alias.asname or alias.name,
                    "line": node.lineno,
                    "symbol": alias.name,
                })

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.variables.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.loaded_names.setdefault(node.id, []).append(node.lineno)

    def visit_Attribute(self, node):
        path = _call_name(node)
        local, separator, member = path.partition(".")
        if separator:
            self.attribute_uses.setdefault(local, []).append((member, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.assignments.append((node, _names_in_target_list(node.targets)))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.assignments.append((node, _names_in_target(node.target)))
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.assignments.append((node, _names_in_target(node.target)))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.functions.append(node)
        self.parameters.update(arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))
        if node.args.vararg:
            self.parameters.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.parameters.add(node.args.kwarg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.classes.append(node)
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops.append(node)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.loops.append(node)
        self.generic_visit(node)

    def visit_If(self, node):
        self.conditions.append(node)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.conditions.append(node)
        self.generic_visit(node)

    def visit_Try(self, node):
        self.tries.append(node)
        self.generic_visit(node)

    visit_TryStar = visit_Try

    def visit_With(self, node):
        self.withs.append(node)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_Call(self, node):
        name = _call_name(node.func)
        self.calls.append((name, node.lineno))
        if name == "print":
            self.print_calls.append(node)
        elif name == "input":
            self.input_calls.append(node)
        self.generic_visit(node)

    def visit_Return(self, node):
        self.returns.append(node)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.binops.append(node)
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.comparisons.append(node)
        self.generic_visit(node)

    def visit_List(self, node):
        self.collections.add("list")
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.collections.add("list comprehension")
        self.comprehensions.append(node)
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.collections.add("dictionary")
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.collections.add("dictionary comprehension")
        self.comprehensions.append(node)
        self.generic_visit(node)

    def visit_Set(self, node):
        self.collections.add("set")
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.collections.add("set comprehension")
        self.comprehensions.append(node)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.collections.add("generator expression")
        self.comprehensions.append(node)
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.collections.add("tuple")
        self.generic_visit(node)


def _names_in_target_list(targets):
    return sorted({name for target in targets for name in _names_in_target(target)})


def _static_value(node, names=None):
    """Evaluate only bounded literal expressions, never names or calls."""
    names = names or {}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_static_value(item, names) for item in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](values)
    if isinstance(node, ast.Dict):
        return {_static_value(key, names): _static_value(value, names) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp):
        value = _static_value(node.operand, names)
        operation = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}.get(type(node.op))
        if operation:
            return operation(value)
    if isinstance(node, ast.BinOp):
        left, right = _static_value(node.left, names), _static_value(node.right, names)
        operations = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }
        operation = operations.get(type(node.op))
        if operation:
            if type(left) is int and type(right) is int and max(left.bit_length(), right.bit_length()) > 4096:
                raise ValueError("literal is too large")
            if type(node.op) is ast.Add and type(left) is type(right) and isinstance(left, (str, list, tuple)):
                if len(left) + len(right) > 1000:
                    raise ValueError("literal is too large")
            if type(node.op) is ast.Mult and (isinstance(left, (str, list, tuple)) or isinstance(right, (str, list, tuple))):
                sequence, count = (left, right) if isinstance(left, (str, list, tuple)) else (right, left)
                if type(count) is int and len(sequence) * max(count, 0) > 1000:
                    raise ValueError("literal is too large")
            result = operation(left, right)
            if type(result) is int and result.bit_length() > 4096:
                raise ValueError("literal result is too large")
            return result
    raise ValueError("value depends on runtime data")


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _call_name(node.value) + "." + node.attr
    return "call"


def _parameters(function):
    args = [*function.args.posonlyargs, *function.args.args]
    names = [arg.arg for arg in args]
    if function.args.vararg:
        names.append("*" + function.args.vararg.arg)
    names.extend(arg.arg for arg in function.args.kwonlyargs)
    if function.args.kwarg:
        names.append("**" + function.args.kwarg.arg)
    return names


def _statement_steps(statements, source, depth=0):
    steps = []
    indent = "  " * depth
    for node in statements:
        line = _line(node)
        prefix = f"{indent}Line {line}: "
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = ", ".join(_parameters(node)) or "no parameters"
            steps.append(prefix + f"Defines function `{node.name}` with {params}. Its body runs when the function is called.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
        elif isinstance(node, ast.ClassDef):
            steps.append(prefix + f"Defines class `{node.name}`. Its body describes the class and runs when the class is created.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            steps.append(prefix + f"Makes imported module or names available: `{_source(node, source)}`.")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names = _names_in_target_list(node.targets) if isinstance(node, ast.Assign) else _names_in_target(node.target)
            target_text = ", ".join(f"`{name}`" for name in names) or "a value"
            value_node = getattr(node, "value", None)
            detail = ""
            if value_node is not None and any(
                isinstance(item, ast.Call) and _call_name(item.func) == "input"
                for item in ast.walk(value_node)
            ):
                detail = " The expression also requests user input."
            if value_node is not None:
                nested_calls = sorted({
                    _call_name(item.func) for item in ast.walk(value_node)
                    if isinstance(item, ast.Call)
                })
                if nested_calls and not (nested_calls == ["input"]):
                    detail += " It evaluates function call(s): " + ", ".join(f"`{name}`" for name in nested_calls) + "."
                for comprehension in (item for item in ast.walk(value_node) if isinstance(item, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))):
                    iterable = _source(comprehension.generators[0].iter, source)
                    detail += f" It builds a {type(comprehension).__name__} by iterating over `{iterable}`."
            steps.append(prefix + f"Stores or updates {target_text} using `{_source(node, source)}`.{detail}")
        elif isinstance(node, ast.If):
            steps.append(prefix + f"Checks `{_source(node.test, source)}` and runs one branch depending on whether it is true.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
            if node.orelse:
                steps.append(indent + "  Otherwise, the alternative branch is considered:")
                steps.extend(_statement_steps(node.orelse, source, depth + 1))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            steps.append(prefix + f"Repeats the loop body for values from `{_source(node.iter, source)}`, assigning each to its loop variable.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
            if node.orelse:
                steps.append(indent + "  If the loop finishes without a break, its else block runs:")
                steps.extend(_statement_steps(node.orelse, source, depth + 1))
        elif isinstance(node, ast.While):
            steps.append(prefix + f"Repeats the body while `{_source(node.test, source)}` remains true.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
            if node.orelse:
                steps.extend(_statement_steps(node.orelse, source, depth + 1))
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            steps.append(prefix + "Runs a protected block and selects an exception handler if a matching error occurs.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
            for handler in node.handlers:
                label = _source(handler.type, source) if handler.type else "any exception"
                steps.append(indent + f"  If `{label}` is raised, the handler runs:")
                steps.extend(_statement_steps(handler.body, source, depth + 1))
            if node.orelse:
                steps.append(indent + "  If no exception occurs, the try-else block runs:")
                steps.extend(_statement_steps(node.orelse, source, depth + 1))
            if node.finalbody:
                steps.append(indent + "  The finally block runs after the try handling:")
                steps.extend(_statement_steps(node.finalbody, source, depth + 1))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            contexts = ", ".join(f"`{_source(item.context_expr, source)}`" for item in node.items)
            steps.append(prefix + f"Enters context(s) managed by {contexts} and runs the body.")
            steps.extend(_statement_steps(node.body, source, depth + 1))
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                name = _call_name(node.value.func)
                if name == "input":
                    steps.append(prefix + "Requests input from the user.")
                elif name == "print":
                    steps.append(prefix + "Displays values with `print()`.")
                else:
                    steps.append(prefix + f"Calls `{name}`.")
            else:
                steps.append(prefix + f"Evaluates `{_source(node.value, source)}`.")
        elif isinstance(node, ast.Return):
            value = f" `{_source(node.value, source)}`" if node.value else " no explicit value"
            steps.append(prefix + f"Returns{value} from the current function.")
        elif isinstance(node, ast.Raise):
            steps.append(prefix + f"Raises `{_source(node.exc, source) if node.exc else 'the active exception'}`.")
        elif isinstance(node, ast.Assert):
            steps.append(prefix + f"Checks the assertion `{_source(node.test, source)}`; a false result raises an error.")
        elif isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
            action = {ast.Break: "Exits the nearest loop", ast.Continue: "Moves to the next loop iteration", ast.Pass: "Does nothing at this point"}[type(node)]
            steps.append(prefix + action + ".")
        elif isinstance(node, ast.Delete):
            steps.append(prefix + f"Deletes the target in `{_source(node, source)}`.")
        elif isinstance(node, ast.Match):
            steps.append(prefix + f"Compares `{_source(node.subject, source)}` with its case patterns and runs a matching case.")
            for case in node.cases:
                steps.extend(_statement_steps(case.body, source, depth + 1))
        else:
            steps.append(prefix + f"Processes a `{type(node).__name__}` statement: `{_source(node, source)}`.")
    return steps


def _library_section(imports, facts):
    if not imports:
        return ["No external libraries are used. The program uses Python's built-in features."]

    rows = []
    for item in imports:
        module = item["module"] or "relative module"
        root = module.lstrip(".").split(".", 1)[0].lower()
        is_stdlib = not module.startswith(".") and root in sys.stdlib_module_names
        symbol = item.get("symbol")
        imported_name = item["local"]
        uses = facts.loaded_names.get(imported_name, [])
        if imported_name == "*":
            usage = "A wildcard import is present, so individual imported names cannot be identified reliably."
        elif facts.attribute_uses.get(imported_name):
            members = sorted(set(member for member, _ in facts.attribute_uses[imported_name]))
            lines = ", ".join(str(line) for _, line in sorted(set(facts.attribute_uses[imported_name]))[:8])
            usage = f"The program uses member(s) `{', '.join(members)}` through this import on line(s) {lines}."
        elif symbol and any(name == symbol for name, _ in facts.calls):
            lines = ", ".join(str(line) for name, line in facts.calls if name == symbol)
            usage = f"The imported function or class `{symbol}` is called on line(s) {lines}."
        elif uses:
            lines = ", ".join(str(line) for line in sorted(set(uses))[:8])
            usage = f"The imported name `{imported_name}` is referenced on line(s) {lines}."
        else:
            usage = "No use of this imported name was found, so its specific role is not clear from the source."

        if is_stdlib:
            kind = "Python standard library"
            purpose = _LIBRARY_PURPOSES.get(root, "is part of Python and provides general-purpose functionality")
        else:
            kind = "External or project module"
            purpose = _LIBRARY_PURPOSES.get(root, "provides functionality defined by that package or project module")
        label = f"`{module}`" + (f" (`{symbol}` as `{imported_name}`)" if symbol else f" as `{imported_name}`")
        rows.append(f"{label} ({kind}): {purpose}. {usage}")
    return rows


def _why_section(facts, source):
    reasons = []
    for node, names in facts.assignments:
        targets = ", ".join(f"`{name}`" for name in names) or "a value"
        later_used = [name for name in names if facts.loaded_names.get(name)]
        if later_used:
            reason = f"Line {_line(node)} assigns or updates {targets}, making those values available to later expressions."
        else:
            reason = f"Line {_line(node)} assigns or updates {targets}, storing a value. The source does not show a later read of that name, so its purpose cannot be determined from the code alone."
        reasons.append(reason)
    for node in facts.conditions:
        condition = node.test if isinstance(node, (ast.If, ast.IfExp)) else node
        reasons.append(f"Line {_line(node)} checks `{_source(condition, source)}` so the program can choose behavior based on a condition.")
    for node in facts.loops:
        purpose = "iterates over items or a range" if isinstance(node, (ast.For, ast.AsyncFor)) else "repeats while a condition remains true"
        reasons.append(f"Line {_line(node)} uses a loop that {purpose}, avoiding repeated copies of the same instructions.")
    for node in facts.functions:
        reasons.append(f"Line {_line(node)} defines `{node.name}` to group instructions into a reusable operation; it runs when called.")
    for node in facts.classes:
        reasons.append(f"Line {_line(node)} defines `{node.name}` as a class for grouping related data and behavior.")
    for node in facts.comprehensions:
        iterable = _source(node.generators[0].iter, source)
        reasons.append(f"Line {_line(node)} uses a comprehension to create a value by processing items from `{iterable}` in a compact form.")
    for node in facts.tries:
        reasons.append(f"Line {_line(node)} uses exception handling to respond to errors that may occur in the protected block.")
    for node in facts.withs:
        reasons.append(f"Line {_line(node)} uses a context manager to manage a resource or temporary context around its block.")
    for node in facts.input_calls:
        reasons.append(f"Line {_line(node)} calls `input()` to obtain a value from the user while the program runs.")
    for node in facts.print_calls:
        reasons.append(f"Line {_line(node)} calls `print()` to display information.")
    for name, line in facts.calls:
        if name not in {"print", "input"}:
            reasons.append(f"Line {line} calls `{name}` to perform an operation. Its exact purpose depends on that function or method's behavior.")
    for node in facts.binops:
        operation = _BINARY_NAMES.get(type(node.op), "an operation")
        reasons.append(f"Line {_line(node)} uses {operation} (`{_source(node, source)}`) to combine or transform values.")
    for node in facts.returns:
        reasons.append(f"Line {_line(node)} returns a value to the code that called the function; returning a value does not display it by itself.")
    if not reasons:
        reasons.append("No assignments, control-flow blocks, function or class definitions, input/output calls, or arithmetic operations were identified. The purpose of the source cannot be inferred beyond its statements.")
    return reasons


def _concepts(facts, imports):
    concepts = []
    if facts.variables or facts.parameters:
        concepts.append("Variables and function parameters")
    concepts.extend(sorted(facts.collections))
    if facts.comprehensions:
        concepts.append("Comprehensions")
    if facts.functions:
        concepts.append("Function definitions")
    if facts.returns:
        concepts.append("Return values")
    if facts.calls:
        concepts.append("Function and method calls")
    if facts.classes:
        concepts.append("Classes")
    if any(isinstance(node, (ast.For, ast.AsyncFor)) for node in facts.loops):
        concepts.append("For loops")
    if any(isinstance(node, ast.While) for node in facts.loops):
        concepts.append("While loops")
    if facts.conditions:
        concepts.append("Conditions")
    if facts.tries:
        concepts.append("Exception handling")
    if facts.withs or any(name == "open" or name.endswith(".open") for name, _ in facts.calls):
        concepts.append("File or resource handling")
    if facts.binops:
        concepts.append("Arithmetic or value operations")
    if facts.comparisons:
        concepts.append("Comparisons")
    if facts.input_calls:
        concepts.append("User input")
    if facts.print_calls:
        concepts.append("Printed output")
    if imports:
        concepts.append("Imported modules")
    return concepts or ["Basic Python statements"]


def _output_section(facts, tree):
    if not facts.print_calls:
        output = ["No direct `print()` calls were found. Returned values are passed back to callers and are not automatically displayed."]
        if not facts.returns:
            output[0] += " Output through files, interfaces, or other side effects cannot be determined from static inspection alone."
        else:
            output.extend(f"Line {_line(node)} returns a value from a function; the value depends on its arguments and is not printed here." for node in facts.returns)
        return output

    print_environments = {}
    known_values = {}
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and _call_name(statement.value.func) == "print":
            print_environments[id(statement.value)] = dict(known_values)
        elif isinstance(statement, ast.Assign):
            try:
                value = _static_value(statement.value, known_values)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
                value = _UNKNOWN
                known_values.clear()
            for target in statement.targets:
                if isinstance(target, ast.Name) and value is not _UNKNOWN:
                    known_values[target.id] = value
                else:
                    if not isinstance(target, ast.Name):
                        known_values.clear()
                    for name in _names_in_target(target):
                        known_values.pop(name, None)
        elif isinstance(statement, ast.AnnAssign):
            if isinstance(statement.target, ast.Name) and statement.value is not None:
                try:
                    known_values[statement.target.id] = _static_value(statement.value, known_values)
                except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
                    known_values.pop(statement.target.id, None)
                    known_values.clear()
        elif isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            expression = ast.BinOp(left=ast.Name(id=statement.target.id, ctx=ast.Load()), op=statement.op, right=statement.value)
            try:
                known_values[statement.target.id] = _static_value(expression, known_values)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
                known_values.pop(statement.target.id, None)
                known_values.clear()
        elif isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known_values.clear()
        elif not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and _call_name(statement.value.func) == "print"):
            known_values.clear()

    output = []
    for node in facts.print_calls:
        line = _line(node)
        environment = print_environments.get(id(node), {})
        try:
            values = [_static_value(arg, environment) for arg in node.args]
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                raise ValueError
            keywords = {item.arg: _static_value(item.value, environment) for item in node.keywords if item.arg in {"sep", "end"}}
            if any(item.arg not in {"sep", "end"} for item in node.keywords):
                raise ValueError
            sep = keywords.get("sep", " ")
            end = keywords.get("end", "\n")
            if not isinstance(sep, str) or not isinstance(end, str):
                raise ValueError
            text = sep.join(str(value) for value in values)
            output.append(f"Line {line}: If reached, this `print()` call displays {text!r} (then uses its `end` value).")
        except (ValueError, TypeError, SyntaxError):
            output.append(f"Line {line}: This `print()` call displays values computed at runtime; the exact text cannot be determined safely from the source alone.")
    if facts.returns:
        output.extend(f"Line {_line(node)} returns a value to its caller; that value is not automatically printed." for node in facts.returns)
    return output


def explain_program(source):
    """Return AST-derived explanation sections without executing source."""
    if not isinstance(source, str) or not source.strip():
        return {"status": "empty", "message": "Enter Python source code to explain."}
    if len(source) > MAX_SOURCE_LENGTH:
        return {"status": "unsupported", "message": f"This source is longer than the current {MAX_SOURCE_LENGTH:,}-character analysis limit."}

    try:
        tree = ast.parse(source)
        node_count = sum(1 for _ in ast.walk(tree))
    except (SyntaxError, ValueError, MemoryError, RecursionError) as error:
        message = error.msg if isinstance(error, SyntaxError) else "The Python structure could not be parsed safely."
        location = f" on line {error.lineno}" if isinstance(error, SyntaxError) and error.lineno else ""
        return {"status": "syntax_error", "message": f"A program explanation could not be generated{location}: {message}"}
    if node_count > MAX_AST_NODES:
        return {"status": "unsupported", "message": f"This source contains more than the current {MAX_AST_NODES:,}-node analysis limit."}

    facts = _Facts()
    try:
        facts.visit(tree)
    except RecursionError:
        return {"status": "unsupported", "message": "This source is nested too deeply for a safe structural explanation."}
    facts.variables.update(facts.parameters)
    try:
        steps = _statement_steps(tree.body, source)
    except RecursionError:
        return {"status": "unsupported", "message": "This source is nested too deeply for a safe step-by-step explanation."}
    actions = []
    if facts.input_calls:
        actions.append("requests user input")
    if facts.assignments:
        actions.append("stores or updates values")
    if facts.functions:
        actions.append("defines functions")
    if facts.classes:
        actions.append("defines classes")
    if facts.loops or facts.comprehensions:
        actions.append("repeats instructions with loops")
    if facts.conditions:
        actions.append("selects behavior with conditions")
    if facts.print_calls:
        actions.append("prints information")
    overview = f"This program has {len(tree.body)} top-level statement(s) and {node_count} Python syntax element(s)."
    if actions:
        overview += " It " + ", ".join(actions) + "."
    else:
        overview += " Its overall purpose cannot be determined reliably from its syntax alone."
    overview += " The explanation is based on source structure; runtime-dependent behavior may differ."

    execution = [
        "Python starts with the module-level statements and processes them in source order.",
        "Function bodies run when their functions are called, not merely when the definitions are read. Class definitions are processed when execution reaches them.",
        *steps,
        "Branches, loops, user input, external data, and runtime state determine which statements actually run and what values they contain.",
    ]
    simple = "The program " + (", ".join(actions) if actions else "contains Python statements whose overall purpose is not clear from syntax alone") + ". Python follows the statements in order, while conditions and loops control which steps happen. "
    simple += "This explanation describes the code structure without running it, so exact results may depend on inputs or runtime data." if facts.input_calls or facts.loops or facts.conditions or facts.calls else "This explanation is based on the written code and does not run it."

    return {
        "status": "ok",
        "sections": {
            "PROGRAM OVERVIEW": [overview],
            "PROGRAM LOGIC": steps or ["No executable statements were found in the module body."],
            "WHY THIS LOGIC IS USED": _why_section(facts, source),
            "LIBRARIES USED": _library_section(facts.imports, facts),
            "IMPORTANT PYTHON CONCEPTS": _concepts(facts, facts.imports),
            "STEP-BY-STEP EXECUTION": execution,
            "OUTPUT": _output_section(facts, tree),
            "SIMPLE EXPLANATION": [simple],
        },
    }