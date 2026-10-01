"""
Generates dataset/errors.csv with diverse, realistic Python error-message
variations across 12 common error categories. Run once:
    python dataset/generate_dataset.py
"""
import csv
import os

VAR_NAMES = ["x", "total", "count", "result", "student", "value", "data",
             "item", "score", "name", "index", "n", "user", "price", "avg"]
MODULES = ["pandas", "numpy", "matplotlib", "requests", "torch", "flask",
           "cv2", "seaborn", "sklearn", "tensorflow", "bs4", "yaml"]
FILES = ["data.csv", "config.json", "report.txt", "model.pkl", "input.txt",
         "settings.ini", "users.db", "notes.md"]
ATTRS = ["append", "split", "upper", "shape", "values", "read", "get",
         "strip", "items", "sort"]
TYPES = ["str", "int", "list", "NoneType", "dict", "float", "tuple"]
KEYS = ["'id'", "'name'", "'email'", "'age'", "'status'", "'price'", "'user'"]

KB = {
    "NameError": dict(
        explanation="Python cannot find the variable or name being used.",
        cause="The variable was used before it was defined or assigned.",
        solution="Define the variable before using it, and check for typos in the name.",
        example="x = 10\nprint(x)"),
    "TypeError": dict(
        explanation="An operation or function was applied to a value of the wrong type.",
        cause="Two incompatible types were combined, or a function got an argument of the wrong type.",
        solution="Check the types of the values involved and convert them if necessary.",
        example="age = int(input('Age: '))\nprint(age + 1)"),
    "ValueError": dict(
        explanation="A function received an argument of the right type but an inappropriate value.",
        cause="The value passed in could not be processed as expected, e.g. converting text to a number.",
        solution="Validate input values before using them, and handle invalid cases explicitly.",
        example="num = int('42')  # works\nnum = int('abc')  # fails"),
    "IndexError": dict(
        explanation="The requested position in a list or sequence does not exist.",
        cause="An index outside the valid range of the list was used.",
        solution="Check the length of the list before accessing an index, or use a valid index.",
        example="numbers = [1, 2, 3]\nprint(numbers[1])"),
    "KeyError": dict(
        explanation="The requested key does not exist in the dictionary.",
        cause="An attempt was made to access a dictionary key that was never added.",
        solution="Check whether the key exists first (e.g. with .get() or 'in'), or add it before use.",
        example="data = {'id': 1}\nprint(data.get('id'))"),
    "SyntaxError": dict(
        explanation="The code does not follow Python's grammar rules.",
        cause="A statement is malformed, such as a missing colon, bracket, or quote.",
        solution="Re-read the flagged line carefully and fix punctuation, indentation, or keywords.",
        example="if x > 0:\n    print(x)"),
    "IndentationError": dict(
        explanation="The code's indentation does not match what Python expects.",
        cause="Spaces and tabs were mixed, or a block was not indented consistently.",
        solution="Use consistent indentation (4 spaces is standard) throughout the block.",
        example="if True:\n    print('correctly indented')"),
    "ModuleNotFoundError": dict(
        explanation="Python cannot find the module you tried to import.",
        cause="The package is not installed in the current environment.",
        solution="Install the missing package using pip, e.g. pip install <package>.",
        example="pip install pandas\nimport pandas as pd"),
    "ImportError": dict(
        explanation="A specific name could not be imported from a module.",
        cause="The module exists but does not contain the requested attribute or function, or there's a circular import.",
        solution="Check the spelling of the imported name and the module's version/documentation.",
        example="from math import sqrt\nprint(sqrt(16))"),
    "ZeroDivisionError": dict(
        explanation="The code attempted to divide a number by zero.",
        cause="The divisor evaluated to zero, which is mathematically undefined.",
        solution="Check that the divisor is not zero before performing the division.",
        example="if divisor != 0:\n    result = total / divisor"),
    "FileNotFoundError": dict(
        explanation="Python could not find the file you tried to open.",
        cause="The file path is incorrect, the file was moved, or it does not exist yet.",
        solution="Double-check the file path and make sure the file exists before opening it.",
        example="import os\nif os.path.exists('data.csv'):\n    open('data.csv')"),
    "AttributeError": dict(
        explanation="The object does not have the attribute or method being called.",
        cause="A method or property was called on an object that does not support it, often due to a typo or wrong type.",
        solution="Check the object's type and available attributes, or use dir(obj) to inspect it.",
        example="text = 'hello'\nprint(text.upper())"),
}


def gen_examples():
    rows = []

    def add(msg, etype):
        k = KB[etype]
        rows.append([msg, etype, k["explanation"], k["cause"], k["solution"], k["example"]])

    # NameError
    for v in VAR_NAMES:
        add(f"NameError: name '{v}' is not defined", "NameError")
        add(f"name '{v}' is not defined", "NameError")
    add("NameError: free variable referenced before assignment", "NameError")
    add("local variable referenced before assignment", "NameError")

    # TypeError
    for t1, t2 in [("int", "str"), ("str", "int"), ("list", "int"), ("NoneType", "int"),
                   ("dict", "list"), ("float", "str")]:
        add(f"TypeError: unsupported operand type(s) for +: '{t1}' and '{t2}'", "TypeError")
    add("TypeError: 'int' object is not callable", "TypeError")
    add("TypeError: 'NoneType' object is not subscriptable", "TypeError")
    add("TypeError: can only concatenate str (not \"int\") to str", "TypeError")
    add("TypeError: missing 1 required positional argument", "TypeError")
    add("TypeError: takes 2 positional arguments but 3 were given", "TypeError")
    add("TypeError: 'list' object is not callable", "TypeError")
    add("object of type 'NoneType' has no len()", "TypeError")
    add("unsupported operand type(s) for *: 'str' and 'str'", "TypeError")

    # ValueError
    add("ValueError: invalid literal for int() with base 10: 'abc'", "ValueError")
    add("ValueError: could not convert string to float: 'xyz'", "ValueError")
    add("ValueError: too many values to unpack (expected 2)", "ValueError")
    add("ValueError: not enough values to unpack (expected 3, got 1)", "ValueError")
    add("ValueError: math domain error", "ValueError")
    add("ValueError: substring not found", "ValueError")
    add("ValueError: invalid literal for int() with base 10: ''", "ValueError")
    add("ValueError: list.remove(x): x not in list", "ValueError")
    add("ValueError: negative dimensions are not allowed", "ValueError")
    add("ValueError: could not convert string to float: 'NaN%'", "ValueError")
    add("ValueError: operands could not be broadcast together", "ValueError")
    add("ValueError: setting an array element with a sequence", "ValueError")
    add("ValueError: empty range for randrange()", "ValueError")

    # IndexError
    add("IndexError: list index out of range", "IndexError")
    add("list index out of range", "IndexError")
    add("IndexError: string index out of range", "IndexError")
    add("IndexError: tuple index out of range", "IndexError")
    add("IndexError: index 5 is out of bounds for axis 0 with size 3", "IndexError")
    add("index out of range in self", "IndexError")
    add("IndexError: pop from empty list", "IndexError")
    add("IndexError: list assignment index out of range", "IndexError")
    add("IndexError: index 10 is out of bounds for axis 1 with size 4", "IndexError")
    add("IndexError: too many indices for array", "IndexError")
    add("IndexError: deque index out of range", "IndexError")

    # KeyError
    for k in KEYS:
        add(f"KeyError: {k}", "KeyError")
    add("KeyError: 0", "KeyError")
    add("KeyError: 'username'", "KeyError")
    add("KeyError: 'total_price'", "KeyError")
    add("KeyError: 'phone_number'", "KeyError")
    add("KeyError: 'address'", "KeyError")
    add("KeyError: 'category'", "KeyError")

    # SyntaxError
    add("SyntaxError: invalid syntax", "SyntaxError")
    add("SyntaxError: unexpected EOF while parsing", "SyntaxError")
    add("SyntaxError: EOL while scanning string literal", "SyntaxError")
    add("SyntaxError: invalid syntax. Perhaps you forgot a comma?", "SyntaxError")
    add("SyntaxError: unmatched ')'", "SyntaxError")
    add("SyntaxError: expected ':'", "SyntaxError")
    add("SyntaxError: cannot assign to literal", "SyntaxError")
    add("SyntaxError: Missing parentheses in call to 'print'", "SyntaxError")
    add("SyntaxError: invalid character in identifier", "SyntaxError")
    add("SyntaxError: unterminated string literal", "SyntaxError")
    add("SyntaxError: positional argument follows keyword argument", "SyntaxError")
    add("SyntaxError: cannot assign to expression here", "SyntaxError")
    add("SyntaxError: multiple statements found while compiling a single statement", "SyntaxError")

    # IndentationError
    add("IndentationError: expected an indented block", "IndentationError")
    add("IndentationError: unexpected indent", "IndentationError")
    add("IndentationError: unindent does not match any outer indentation level", "IndentationError")
    add("IndentationError: expected an indented block after 'if' statement", "IndentationError")
    add("inconsistent use of tabs and spaces in indentation", "IndentationError")
    add("IndentationError: expected an indented block after function definition", "IndentationError")
    add("IndentationError: unexpected unindent", "IndentationError")
    add("IndentationError: expected an indented block after class definition", "IndentationError")
    add("IndentationError: expected an indented block after 'for' statement", "IndentationError")
    add("TabError: inconsistent use of tabs and spaces", "IndentationError")

    # ModuleNotFoundError
    for m in MODULES:
        add(f"ModuleNotFoundError: No module named '{m}'", "ModuleNotFoundError")
    add("No module named 'pandas'", "ModuleNotFoundError")
    add("ModuleNotFoundError: No module named 'src'", "ModuleNotFoundError")

    # ImportError
    add("ImportError: cannot import name 'sqrt' from 'math'", "ImportError")
    add("ImportError: attempted relative import with no known parent package", "ImportError")
    add("ImportError: cannot import name 'X' from partially initialized module", "ImportError")
    add("ImportError: DLL load failed while importing", "ImportError")
    add("cannot import name 'train_test_split' from 'sklearn'", "ImportError")
    add("ImportError: No module named 'six'", "ImportError")
    add("ImportError: cannot import name 'joblib' from 'sklearn.externals'", "ImportError")
    add("ImportError: numpy.core.multiarray failed to import", "ImportError")
    add("ImportError: cannot import name 'app' from partially initialized module 'main'", "ImportError")

    # ZeroDivisionError
    add("ZeroDivisionError: division by zero", "ZeroDivisionError")
    add("division by zero", "ZeroDivisionError")
    add("ZeroDivisionError: float division by zero", "ZeroDivisionError")
    add("ZeroDivisionError: integer division or modulo by zero", "ZeroDivisionError")
    add("ZeroDivisionError: float floor division by zero", "ZeroDivisionError")
    for v in ["total", "average", "ratio", "score", "result", "count"]:
        add(f"ZeroDivisionError: {v} / 0", "ZeroDivisionError")
    add("ZeroDivisionError: modulo by zero", "ZeroDivisionError")
    add("cannot divide by zero", "ZeroDivisionError")
    add("ZeroDivisionError raised while computing average", "ZeroDivisionError")

    # FileNotFoundError
    for f in FILES:
        add(f"FileNotFoundError: [Errno 2] No such file or directory: '{f}'", "FileNotFoundError")
    add("FileNotFoundError: could not find file", "FileNotFoundError")
    add("FileNotFoundError: [Errno 2] No such file or directory: 'log.txt'", "FileNotFoundError")
    add("FileNotFoundError: Unable to open file", "FileNotFoundError")
    add("FileNotFoundError: image.png not found", "FileNotFoundError")
    add("FileNotFoundError: No such directory: 'output/'", "FileNotFoundError")

    # AttributeError
    for a in ATTRS:
        add(f"AttributeError: 'NoneType' object has no attribute '{a}'", "AttributeError")
    add("AttributeError: 'list' object has no attribute 'append_item'", "AttributeError")
    add("AttributeError: module 'numpy' has no attribute 'array2'", "AttributeError")
    add("AttributeError: 'DataFrame' object has no attribute 'sortvalues'", "AttributeError")

    return rows


def main():
    rows = gen_examples()
    out_path = os.path.join(os.path.dirname(__file__), "errors.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["error_message", "error_type", "explanation", "cause", "solution", "example"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    # quick category distribution
    from collections import Counter
    counts = Counter(r[1] for r in rows)
    for k, v in sorted(counts.items()):
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
