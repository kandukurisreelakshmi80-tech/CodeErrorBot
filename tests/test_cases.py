"""
test_cases.py
-------------
Runs the trained model against error messages NOT present in the
training dataset (Phase 19-20) and reports Input / Expected / Predicted
/ Correct / Confidence for each, plus an overall accuracy summary.

Run: python tests/test_cases.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predictor import get_predictor  # noqa: E402

# 27 unseen test cases spanning every supported category.
TEST_CASES = [
    ("NameError: name 'student' is not defined", "NameError"),
    ("name 'config' is not defined", "NameError"),
    ("NameError: name 'output' is not defined", "NameError"),
    ("TypeError: unsupported operand type(s) for -: 'str' and 'int'", "TypeError"),
    ("TypeError: 'float' object is not iterable", "TypeError"),
    ("argument of type 'int' is not iterable", "TypeError"),
    ("ValueError: invalid literal for int() with base 10: 'twelve'", "ValueError"),
    ("ValueError: too many values to unpack (expected 4)", "ValueError"),
    ("IndexError: list index out of range", "IndexError"),
    ("IndexError: string index out of range while parsing", "IndexError"),
    ("KeyError: 'user_id'", "KeyError"),
    ("KeyError: 'discount_code'", "KeyError"),
    ("SyntaxError: invalid syntax near if statement", "SyntaxError"),
    ("SyntaxError: unexpected EOF while parsing function", "SyntaxError"),
    ("IndentationError: expected an indented block after while statement", "IndentationError"),
    ("IndentationError: unindent does not match outer level", "IndentationError"),
    ("ModuleNotFoundError: No module named 'numpy'", "ModuleNotFoundError"),
    ("No module named 'scipy'", "ModuleNotFoundError"),
    ("ImportError: cannot import name 'load_model' from 'keras'", "ImportError"),
    ("ImportError: attempted relative import beyond top-level package", "ImportError"),
    ("ZeroDivisionError: division by zero", "ZeroDivisionError"),
    ("ZeroDivisionError: float division by zero in average calc", "ZeroDivisionError"),
    ("FileNotFoundError: No such file or directory: 'results.json'", "FileNotFoundError"),
    ("FileNotFoundError: Unable to locate dataset.csv", "FileNotFoundError"),
    ("AttributeError: 'NoneType' object has no attribute 'lower'", "AttributeError"),
    ("AttributeError: 'dict' object has no attribute 'append'", "AttributeError"),
    ("AttributeError: module 'os' has no attribute 'pathh'", "AttributeError"),
]


def run_tests():
    predictor = get_predictor()
    correct = 0
    rows = []

    for message, expected in TEST_CASES:
        result = predictor.predict(message)
        predicted = result["predicted_type"]
        confidence = result["confidence"]
        is_correct = predicted == expected
        correct += int(is_correct)
        rows.append((message, expected, predicted, is_correct, confidence))

    print(f"{'Input':<55}{'Expected':<20}{'Predicted':<20}{'OK':<5}{'Conf.'}")
    print("-" * 115)
    for message, expected, predicted, is_correct, confidence in rows:
        mark = "✅" if is_correct else "❌"
        print(f"{message[:53]:<55}{expected:<20}{predicted:<20}{mark:<5}{confidence:.2f}")

    total = len(rows)
    acc = correct / total if total else 0
    print("-" * 115)
    print(f"Overall accuracy on unseen test cases: {correct}/{total} = {acc * 100:.1f}%")
    return rows, acc


if __name__ == "__main__":
    run_tests()
