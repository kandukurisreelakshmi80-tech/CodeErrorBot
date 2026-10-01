"""Regression tests for source-code diagnostics."""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from code_analyzer import (  # noqa: E402
    analyze_source,
)


ERROR_CASES = [
    ("if x > 10\n    print(x)", "SyntaxError", 1),
    ("print(username)", "NameError", 1),
    ('age = "20"\nprint(age + 5)', "TypeError", 2),
    ('number = int("hello")', "ValueError", 1),
    ("numbers = [1, 2, 3]\nprint(numbers[5])", "IndexError", 2),
    ('student = {"name": "Rama"}\nprint(student["age"])', "KeyError", 2),
    ('if True:\nprint("unindented")', "IndentationError", 2),
    ("print(10 / 0)", "ZeroDivisionError", 1),
    ('name = "Rama"\nname.append("M")', "AttributeError", 2),
]


class SourceAnalysisTests(unittest.TestCase):
    def test_required_error_types_and_locations(self):
        for source, expected_type, expected_line in ERROR_CASES:
            with self.subTest(error_type=expected_type):
                result = analyze_source(source)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], expected_type)
                self.assertEqual(result["line"], expected_line)
                self.assertTrue(result["problematic_code"])

    def test_name_error_guidance_avoids_generic_message_and_guessed_value(self):
        from explanation import get_engine

        result = get_engine().analyze_source("print(age)")
        issue = result["errors"][0]
        self.assertEqual(issue["error_type"], "NameError")
        self.assertEqual(issue["line"], 1)
        self.assertIn("age", issue["context_explanation"])
        self.assertTrue(issue["hint"])
        self.assertNotIn("corrected_code", result)

    def test_multiple_errors_keep_source_and_never_return_corrected_code(self):
        source = "print(age)\nprint('20' + 5)"
        result = analyze_source(source)
        self.assertEqual([issue["error_type"] for issue in result["errors"]], ["NameError", "TypeError"])
        self.assertEqual([issue["problematic_code"] for issue in result["errors"]], source.splitlines())
        self.assertEqual(result["source"], source)
        for item in [result, *result["errors"]]:
            for field in ("corrected_code", "correction_status", "changes", "replacement", "solution", "example"):
                self.assertNotIn(field, item)

        from explanation import get_engine

        explained = get_engine().analyze_source(source)
        self.assertEqual([issue["error_type"] for issue in explained["errors"]], ["NameError", "TypeError"])
        for issue in explained["errors"]:
            self.assertTrue(issue["explanation"])
            self.assertTrue(issue["hint"])
            self.assertEqual(issue["problematic_code"], source.splitlines()[issue["line"] - 1])

    def test_every_supported_error_has_explanation_and_hint(self):
        from explanation import get_engine

        engine = get_engine()
        for source, expected_type, _ in ERROR_CASES:
            with self.subTest(error_type=expected_type):
                result = engine.analyze_source(source)
                self.assertEqual(result["error_type"], expected_type)
                self.assertTrue(result["explanation"])
                self.assertTrue(result["cause"])
                self.assertTrue(result["hint"])
                self.assertNotIn("corrected_code", result)

    def test_clean_program_and_empty_input(self):
        clean = analyze_source("value = 2\nprint(value)")
        self.assertEqual(clean["status"], "ok")
        self.assertEqual(clean["line_count"], 2)
        self.assertEqual(analyze_source("  ")["status"], "empty")

    def test_long_source_is_not_truncated(self):
        source = "".join(f"value_{number} = {number}\n" for number in range(3000))
        result = analyze_source(source)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["line_count"], 3000)

    def test_missing_import_names_install_command(self):
        source = (
            "import cv2 as image, sklearn\n"
            "from PIL import Image\n"
            "import codeerrorbot_missing_one, codeerrorbot_missing_two\n"
        )
        with patch("code_analyzer._find_module_spec", return_value=None):
            result = analyze_source(source)
        self.assertEqual(result["status"], "missing_dependencies")
        missing = {item["module"]: item["install_command"] for item in result["missing_dependencies"]}
        self.assertEqual(missing["cv2"], "python -m pip install opencv-python")
        self.assertEqual(missing["sklearn"], "python -m pip install scikit-learn")
        self.assertEqual(missing["PIL"], "python -m pip install Pillow")
        self.assertEqual(missing["codeerrorbot_missing_one"], "python -m pip install codeerrorbot_missing_one")
        self.assertEqual(missing["codeerrorbot_missing_two"], "python -m pip install codeerrorbot_missing_two")

    def test_standard_library_imports_are_not_reported_missing(self):
        result = analyze_source(
            "import math, random, os, sys, json, re\n"
            "from datetime import date\n"
            "print(math.sqrt(4))"
        )
        self.assertNotEqual(result["status"], "missing_dependencies")
        self.assertFalse(result.get("missing_dependencies"))
        self.assertTrue(all(item["kind"] == "standard library" for item in result["import_checks"]))
        self.assertTrue(all(item["status"] == "available" for item in result["import_checks"]))

    def test_unavailable_standard_library_module_has_no_pip_command(self):
        with patch("code_analyzer._find_module_spec", return_value=None):
            result = analyze_source("import tkinter")
        self.assertNotEqual(result["status"], "missing_dependencies")
        self.assertEqual(result["import_checks"][0]["kind"], "standard library")
        self.assertEqual(result["import_checks"][0]["status"], "not found")
        self.assertNotIn("install_command", result["import_checks"][0])

    def test_additional_runtime_errors_have_explanations(self):
        from explanation import get_engine

        cases = [
            ("assert False", "AssertionError", 1),
            ("def read_before_assigning():\n    print(value)\n    value = 1\nread_before_assigning()", "UnboundLocalError", 2),
            ("def recurse():\n    return recurse()\nrecurse()", "RecursionError", 2),
        ]
        engine = get_engine()
        for source, expected_type, expected_line in cases:
            with self.subTest(error_type=expected_type):
                result = engine.analyze_source(source)
                self.assertEqual(result["error_type"], expected_type)
                self.assertEqual(result["line"], expected_line)
                self.assertTrue(result["explanation"])
                self.assertTrue(result["hint"])

    def test_imports_in_comments_and_strings_are_ignored(self):
        result = analyze_source("# import codeerrorbot_missing\ntext = 'import also_missing'\nprint(text)")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["import_checks"], [])

    def test_file_access_is_not_probed_or_executed(self):
        result = analyze_source('open("codeerrorbot_missing_file_xyz.txt")')
        self.assertEqual(result["status"], "unsupported")
        self.assertIn("not inspected or performed", result["message"])

    def test_pyflakes_warnings_are_not_reported_as_confirmed_runtime_errors(self):
        result = analyze_source("import math\nprint(unknown_name)")
        if result.get("dependency_notice"):
            self.assertIn("pip install pyflakes", result["dependency_notice"])
            return
        self.assertEqual(result["status"], "unsupported")
        undefined = [item for item in result["warnings"] if item["category"] == "UndefinedName"]
        self.assertTrue(undefined)
        self.assertIn("not defined", undefined[0]["explanation"])
        required_fields = {"category", "line", "column", "code", "message", "explanation"}
        self.assertTrue(all(required_fields.issubset(warning) for warning in result["warnings"]))

    def test_logic_risks_and_unreachable_code_are_warnings(self):
        source = "def inspect(value):\n    if value == value:\n        return value\n        print('unreachable')"
        result = analyze_source(source)
        self.assertEqual(result["status"], "ok")
        warnings = result["warnings"]
        self.assertTrue(any(item["category"] == "SuspiciousCondition" for item in warnings))
        self.assertTrue(any(item["category"] == "UnreachableCode" for item in warnings))
        self.assertNotIn("errors", result)

    def test_missing_pyflakes_keeps_source_analysis_available(self):
        with patch.dict(sys.modules, {"pyflakes.api": None}):
            result = analyze_source("value = 2")
        self.assertEqual(result["status"], "ok")
        self.assertIn("pip install pyflakes", result["dependency_notice"])

    def test_source_analysis_survives_model_loading_failure(self):
        from explanation import ExplanationEngine

        with patch("explanation.get_predictor", side_effect=RuntimeError("incompatible model")):
            engine = ExplanationEngine()
        result = engine.analyze_source("print(missing_value)")
        self.assertEqual(result["error_type"], "NameError")
        self.assertNotIn("ml_predicted_type", result)
        self.assertEqual(engine.explain("NameError: name 'x' is not defined")["status"], "model_unavailable")

    def test_runaway_nested_loops_stop_at_configured_timeout(self):
        lines = []
        for level in range(7):
            lines.append("    " * level + f"for loop_{level} in range(100):")
        lines.append("    " * 7 + "pass")
        result = analyze_source("\n".join(lines), timeout=0.2)
        self.assertEqual(result["status"], "timeout")

    def test_multiple_undefined_names_are_reported(self):
        result = analyze_source("print(first_missing)\nprint(second_missing)")
        self.assertEqual(result["error_type"], "NameError")
        self.assertEqual(result["additional_issues"][0]["line"], 2)

    def test_unbounded_while_is_rejected_but_static_analysis_continues(self):
        result = analyze_source("while True:\n    pass")
        self.assertEqual(result["status"], "unsupported")

    def test_statically_bounded_while_is_supported(self):
        source = "items = []\ni = 0\nwhile i < 3:\n    items.append(i)\n    i += 1\nprint(items)"
        result = analyze_source(source)
        self.assertEqual(result["status"], "ok")

    def test_functions_loops_and_augmented_assignments_are_supported(self):
        source = "def double(value):\n    return value * 2\ntotal = 0\nfor value in [1, 2, 3]:\n    total += double(value)\nprint(total)"
        result = analyze_source(source)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["functions"], 1)

    def test_requested_function_program_reports_both_errors_without_correction(self):
        source = '''students = []

def add_student(name, age, marks):
    student = {"name": name, "age": age, "marks": marks}
    students.append(student)

def calculate_average(marks):
    total = sum(marks)
    average = total / len(marks)
    return average

def display_student(student):
    print("\\nStudent Details")
    print("----------------")
    print("Name:", student["name"])
    print("Age:", student["age"])
    print("Marks:", student["marks"])

    average = calculate_average(student["marks"])
    print("Average:", average)

def find_topper():
    highest_average = 0
    topper = None
    for student in students:
        average = calculate_average(student["marks"])
        if average > highest_average:
            highest_average = average
            topper = student
    return topper

add_student("Rahul", 20, [85, 90, 78])
add_student("Anjali", 21, [92, 88, 95])
add_student("Kiran", 19, [70, 75, 80])
for student in students:
    display_student(student)

topper = find_topper()
print("\\nTopper:")
print(topper["name"])
print("Average:", calculate_average(topper["marks"]))

print("\\nTotal Students:", student_count)
print("Fourth Student:", students[3])
'''
        result = analyze_source(source)
        self.assertEqual([item["error_type"] for item in result["errors"]], ["NameError", "IndexError"])
        self.assertEqual(result["source"], source)
        self.assertNotIn("corrected_code", result)
        from explanation import get_engine

        explained = get_engine().analyze_source(source)
        self.assertEqual(len(explained["errors"]), 2)
        self.assertEqual(explained["errors"][0]["ml_predicted_type"], "NameError")
        self.assertTrue(explained["errors"][0]["hint"])
        self.assertEqual(explained["errors"][1]["ml_predicted_type"], "IndexError")
        self.assertTrue(explained["errors"][1]["hint"])

    def test_nested_functions_are_supported(self):
        source = "def outer(value):\n    def inner():\n        return value + 1\n    return inner()\nresult = outer(4)"
        result = analyze_source(source)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["functions"], 2)

    def test_dangerous_dynamic_call_is_blocked(self):
        result = analyze_source("eval('1 + 1')")
        self.assertEqual(result["status"], "unsupported")

    def test_nlp_and_actual_code_context_are_combined(self):
        from explanation import get_engine

        result = get_engine().analyze_source("numbers = [10, 20, 30]\nprint(numbers[5])")
        self.assertEqual(result["error_type"], "IndexError")
        self.assertEqual(result["ml_predicted_type"], "IndexError")
        self.assertIn("3 elements", result["context_explanation"])
        self.assertIn("explanation", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)