"""UI result rendering tests for incomplete analyzer metadata."""
import os
import sys
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class AppResultRenderingTests(unittest.TestCase):
    def render_result(self, result):
        app = AppTest.from_file(APP_PATH)
        app.session_state["analysis_result"] = result
        app.run(timeout=30)
        self.assertFalse(app.exception, [error.message for error in app.exception])
        return app

    def test_warnings_with_missing_fields_and_malformed_entries(self):
        app = self.render_result({
            "status": "ok",
            "message": "Analysis completed.",
            "line_count": 1,
            "variables": [],
            "warnings": [
                {
                    "category": "UndefinedName",
                    "line": 2,
                    "message": "undefined name x",
                    "code": "print(x)",
                    "explanation": "The name x has not been defined.",
                },
                {"category": "UnusedImport", "message": "unused import json"},
                {"line": 4},
                {},
                None,
                "malformed warning",
            ],
        })
        warning_text = [warning.value for warning in app.warning]
        self.assertTrue(any("The name x has not been defined" in text for text in warning_text))
        self.assertTrue(any("Location unavailable" in text for text in warning_text))
        self.assertTrue(any("malformed" in text for text in warning_text))

    def test_missing_and_empty_warning_fields(self):
        base_result = {
            "status": "ok",
            "message": "Analysis completed.",
            "line_count": 0,
            "variables": [],
        }
        for warnings in (None, []):
            with self.subTest(warnings=warnings):
                result = {**base_result, "warnings": warnings} if warnings is not None else dict(base_result)
                self.render_result(result)

    def test_incomplete_multiple_error_result_renders_safely(self):
        app = self.render_result({
            "status": "error",
            "errors": [
                {
                    "error_type": "NameError",
                    "line": 1,
                    "problematic_code": "print(age)",
                    "error_message": "NameError: name 'age' is not defined",
                    "context_explanation": "`age` is used but not defined before this line.",
                    "hint": "Check the name's spelling and scope.",
                },
                {
                    "error_type": "TypeError",
                    "line": 2,
                    "problematic_code": "print('20' + 5)",
                    "error_message": "TypeError: invalid operation",
                    "explanation": "These values have incompatible types.",
                    "hint": "Check the types of the values used together on this line.",
                },
            ],
            "source": "print(age)\nprint('20' + 5)",
        })
        self.assertTrue(any("Error 1: `NameError`" in item.value for item in app.markdown))
        self.assertTrue(any("Error 2: `TypeError`" in item.value for item in app.markdown))
        self.assertEqual(sum("**Hint**" in item.value for item in app.markdown), 2)
        self.assertEqual(len(app.code), 0)
        self.assertTrue(any("print(age)" in item.value for item in app.text))
        rendered = "\n".join(item.value for item in [*app.markdown, *app.text])
        self.assertIn("Check the name's spelling and scope.", rendered)
        self.assertIn("Check the types of the values used together on this line.", rendered)

    def test_actual_name_error_shows_specific_guidance_without_generic_text(self):
        from explanation import get_engine

        result = get_engine().analyze_source("print(age)")
        app = self.render_result(result)
        self.assertIn("age", result["errors"][0]["context_explanation"])
        self.assertTrue(any("**Hint**" in item.value for item in app.markdown))
        self.assertNotIn("corrected_code", result)

    def test_assistant_chat_is_not_rendered(self):
        app = AppTest.from_file(APP_PATH).run(timeout=30)
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertEqual(len(app.chat_input), 0)
        self.assertFalse(any("CodeErrorBot Assistant" in item.value for item in app.subheader))
        self.assertNotIn("assistant_chat_history", app.session_state)

    def test_removed_tools_are_not_rendered(self):
        app = AppTest.from_file(APP_PATH)
        app.run(timeout=30)
        self.assertFalse(app.exception, [error.message for error in app.exception])
        button_labels = {button.label for button in app.button}
        self.assertFalse(button_labels & {"Generate starter cases", "Run test cases", "Run property checks"})
        self.assertNotIn("test_case_json", app.session_state)
        self.assertNotIn("property_examples", app.session_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)