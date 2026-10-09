"""Tests for safe, AST-based program explanations."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from program_explainer import explain_program  # noqa: E402


class ProgramExplanationTests(unittest.TestCase):
    def test_sections_reflect_functions_loops_conditions_input_and_output(self):
        source = (
            "def total(values):\n"
            "    result = 0\n"
            "    for value in values:\n"
            "        if value > 0:\n"
            "            result += value\n"
            "    return result\n"
            "amount = input('Count? ')\n"
            "print(total([1, 2, 3]))\n"
            "print(2 + 3)\n"
        )
        result = explain_program(source)
        self.assertEqual(result["status"], "ok")
        sections = result["sections"]
        self.assertEqual(set(sections), {
            "PROGRAM OVERVIEW", "PROGRAM LOGIC", "WHY THIS LOGIC IS USED",
            "LIBRARIES USED", "IMPORTANT PYTHON CONCEPTS",
            "STEP-BY-STEP EXECUTION", "OUTPUT", "SIMPLE EXPLANATION",
        })
        concepts = " ".join(sections["IMPORTANT PYTHON CONCEPTS"])
        self.assertIn("Function definitions", concepts)
        self.assertIn("For loops", concepts)
        self.assertIn("Conditions", concepts)
        self.assertIn("User input", concepts)
        self.assertIn("Count?", " ".join(sections["PROGRAM LOGIC"]))
        output = " ".join(sections["OUTPUT"])
        self.assertIn("5", output)
        self.assertIn("cannot be determined safely", output)

    def test_library_detection_uses_imported_alias_and_call_site(self):
        result = explain_program("import math as m\nprint(m.sqrt(9))")
        library_lines = " ".join(result["sections"]["LIBRARIES USED"])
        self.assertIn("math", library_lines)
        self.assertIn("standard library", library_lines)
        self.assertIn("sqrt", library_lines)
        self.assertIn("line(s) 2", library_lines)
        self.assertNotIn("Pandas", library_lines)

    def test_static_output_uses_prior_literal_assignments(self):
        result = explain_program("answer = 40 + 2\nprint('Answer:', answer)")
        self.assertIn("Answer: 42", " ".join(result["sections"]["OUTPUT"]))

    def test_comprehensions_are_explained_as_iteration(self):
        result = explain_program("doubled = [value * 2 for value in values if value > 0]")
        concepts = " ".join(result["sections"]["IMPORTANT PYTHON CONCEPTS"])
        self.assertIn("Comprehensions", concepts)
        self.assertIn("iterating over `values`", " ".join(result["sections"]["PROGRAM LOGIC"]))

    def test_no_imports_reports_builtin_features_and_only_present_concepts(self):
        result = explain_program("number = 4\nprint(number)")
        self.assertEqual(
            result["sections"]["LIBRARIES USED"],
            ["No external libraries are used. The program uses Python's built-in features."],
        )
        concepts = " ".join(result["sections"]["IMPORTANT PYTHON CONCEPTS"])
        self.assertIn("Variables", concepts)
        self.assertNotIn("Loops", concepts)
        self.assertNotIn("Classes", concepts)

    def test_explainer_never_executes_submitted_source(self):
        marker = os.path.join(os.path.dirname(__file__), "program_explainer_marker.tmp")
        if os.path.exists(marker):
            os.remove(marker)
        result = explain_program(f"open({marker!r}, 'w').write('executed')")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(os.path.exists(marker))

    def test_syntax_error_is_reported_without_running_source(self):
        result = explain_program("if True\n    print('no')")
        self.assertEqual(result["status"], "syntax_error")
        self.assertIn("line 1", result["message"])

    def test_large_program_is_analyzed_with_each_statement(self):
        source = "\n".join(f"value_{index} = {index}" for index in range(2500))
        result = explain_program(source)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["sections"]["PROGRAM LOGIC"]), 2500)


if __name__ == "__main__":
    unittest.main(verbosity=2)