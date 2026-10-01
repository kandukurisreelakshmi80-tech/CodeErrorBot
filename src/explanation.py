"""
explanation.py
---------------
Combines:
    Predicted Error Type + Confidence + Knowledge Base + Optional Similar Error
into a single beginner-friendly explanation object used by app.py.
"""
import json
import os
import re
import sys
import ast

sys.path.append(os.path.dirname(__file__))
from predictor import get_predictor  # noqa: E402
from code_analyzer import analyze_source  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
KB_PATH = os.path.join(BASE_DIR, "dataset", "explanations.json")
ADDITIONAL_ERROR_EXPLANATIONS = {
    "AssertionError": {
        "explanation": "An assert statement's condition was false.",
        "cause": "The program expected a condition to be true, but the value did not meet that expectation.",
    },
    "UnboundLocalError": {
        "explanation": "A function tried to read a local variable before assigning it.",
        "cause": "An assignment makes the name local to the function, so an earlier read cannot use the outer value.",
    },
    "RecursionError": {
        "explanation": "A function called itself too many times without finishing.",
        "cause": "The recursive calls did not reach a stopping condition, or the input requires too many nested calls.",
    },
}


ERROR_HINTS = {
    "NameError": "Check the name's spelling and scope, and whether it was assigned before this line.",
    "UnboundLocalError": "Check where the function assigns this variable and whether that happens before it is read.",
    "TypeError": "Check the types of the values used together on this line.",
    "ValueError": "Check whether the value has the format or range this operation expects.",
    "IndexError": "Check the sequence length and whether the requested position exists.",
    "KeyError": "Check which keys are present in the dictionary at this point.",
    "ZeroDivisionError": "Check whether the divisor can be zero for this input.",
    "AttributeError": "Check the value and type of the object before this attribute access.",
    "SyntaxError": "Review this line and the line before it for missing or misplaced Python syntax.",
    "IndentationError": "Compare this line's indentation with the surrounding block.",
    "AssertionError": "Check which part of the condition is false for the current values.",
    "RecursionError": "Check that each recursive call moves toward a stopping condition.",
    "ModuleNotFoundError": "Check the module name and whether the package is available in this environment.",
    "ImportError": "Check that the imported name exists in the installed module version.",
    "FileNotFoundError": "Check the path and working directory, and whether the file exists there.",
}


def load_knowledge_base(path: str = KB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ExplanationEngine:
    def __init__(self):
        try:
            self.predictor = get_predictor()
            self.model_load_error = None
        except Exception as error:
            self.predictor = None
            self.model_load_error = error
        self.kb = load_knowledge_base()

    def explain(self, error_message: str) -> dict:
        if self.predictor is None:
            return {
                "status": "model_unavailable",
                "message": "The traceback classifier is unavailable. Python source analysis remains available without the saved model.",
            }
        result = self.predictor.predict(error_message)

        if result.get("empty_input"):
            return {
                "status": "empty",
                "message": "Please enter an error message.",
            }

        predicted_type = result["predicted_type"]
        confidence = result["confidence"]
        is_confident = result["is_confident"]

        if not is_confident:
            # Phase 17: handle unknown / low-confidence errors safely.
            closest = result["similar_matches"][0] if result["similar_matches"] else None
            return {
                "status": "uncertain",
                "message": (
                    "Unable to confidently identify this error.\n\n"
                    "Please provide:\n"
                    "• The complete error message\n"
                    "• The complete Python traceback, if available"
                ),
                "predicted_type": predicted_type,
                "confidence": confidence,
                "closest_match": closest,
            }

        info = self.kb.get(predicted_type)
        if info is None:
            # Predicted a label with no KB entry (shouldn't normally happen).
            return {
                "status": "unknown_category",
                "message": f"Identified as '{predicted_type}', but no explanation "
                           f"is available for this category yet.",
                "predicted_type": predicted_type,
                "confidence": confidence,
            }

        return {
            "status": "ok",
            "predicted_type": predicted_type,
            "confidence": confidence,
            "explanation": info["explanation"],
            "cause": info["cause"],
            "hint": ERROR_HINTS.get(predicted_type, "Review the values and operation that triggered this error."),
        }

    def analyze_source(self, source: str) -> dict:
        result = analyze_source(source)
        if result["status"] != "error":
            return result

        info = self.kb.get(result["error_type"]) or ADDITIONAL_ERROR_EXPLANATIONS.get(result["error_type"])
        enriched = {
            **result,
            "explanation": info["explanation"] if info else "Python raised this error while analyzing the submitted program.",
            "cause": info["cause"] if info else "Review the failing operation and the values it uses.",
            "hint": ERROR_HINTS.get(result["error_type"], "Review the values and operation on this line to find what triggered the error."),
        }
        if self.predictor is not None:
            model_result = self.predictor.predict(result["error_message"])
            enriched.update({
                "ml_predicted_type": model_result["predicted_type"],
                "ml_confidence": model_result["confidence"],
                "ml_matches_runtime": model_result["predicted_type"] == result["error_type"],
            })

        if result["error_type"] == "IndexError":
            index_match = re.search(r"\[\s*(\d+)\s*\]", result["problematic_code"])
            list_match = re.search(r"\b([A-Za-z_]\w*)\s*=\s*(\[[^\n]*\])", source)
            if index_match and list_match:
                try:
                    values = ast.literal_eval(list_match.group(2))
                except (ValueError, SyntaxError):
                    values = None
                if isinstance(values, list):
                    index = int(index_match.group(1))
                    enriched["context_explanation"] = (
                        f"The list `{list_match.group(1)}` contains {len(values)} elements, "
                        f"so its valid indexes are 0 through {len(values) - 1}. "
                        f"This line requests index {index}, which is outside that range."
                    )
            if "context_explanation" not in enriched:
                loop_match = re.search(r"for\s+(\w+)\s+in\s+range\((\d+)\)", source)
                if loop_match and list_match:
                    enriched["context_explanation"] = (
                        f"The loop visits {loop_match.group(2)} indexes, but `{list_match.group(1)}` "
                        f"has only {len(ast.literal_eval(list_match.group(2)))} elements. "
                        "The loop reaches an index that the list does not contain."
                    )
        elif result["error_type"] == "NameError":
            name_match = re.search(r"name '([^']+)' is not defined", result["error_message"])
            if name_match:
                identifier = name_match.group(1)
                enriched["context_explanation"] = (
                    f"`{identifier}` is used at line {result['line']}, but it is not assigned "
                    "anywhere in the submitted program before that use."
                )

        detailed_errors = []
        for issue in enriched.get("errors", [enriched]):
            issue_info = self.kb.get(issue["error_type"]) or ADDITIONAL_ERROR_EXPLANATIONS.get(issue["error_type"])
            detailed = {
                **issue,
                "explanation": issue_info["explanation"] if issue_info else "Python raised this error while analyzing the submitted program.",
                "cause": issue_info["cause"] if issue_info else "Review the failing operation and the values it uses.",
                "hint": ERROR_HINTS.get(issue["error_type"], "Review the values and operation on this line to find what triggered the error."),
            }
            if self.predictor is not None:
                issue_model = self.predictor.predict(issue["error_message"])
                detailed.update({
                    "ml_predicted_type": issue_model["predicted_type"],
                    "ml_confidence": issue_model["confidence"],
                    "ml_matches_runtime": issue_model["predicted_type"] == issue["error_type"],
                })
            if issue["error_type"] == "NameError":
                undefined = re.search(r"name '([^']+)' is not defined", issue["error_message"])
                if undefined:
                    name = undefined.group(1)
                    detailed["context_explanation"] = f"`{name}` is read at this line but was not defined earlier in the program."
            elif issue["error_type"] == "IndexError":
                index_match = re.search(r"\b([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]", issue["problematic_code"])
                if index_match:
                    variable, index = index_match.groups()
                    collection = issue["collections"].get(variable)
                    if collection:
                        valid_indexes = max(collection["length"] - 1, 0)
                        detailed["context_explanation"] = (
                            f"`{variable}` currently contains {collection['length']} elements, with valid indexes "
                            f"0 through {valid_indexes}. Index {index} does not exist."
                        )
            detailed_errors.append(detailed)
        enriched["errors"] = detailed_errors
        if detailed_errors:
            for field in ("explanation", "cause", "hint", "context_explanation"):
                if field in detailed_errors[0]:
                    enriched[field] = detailed_errors[0][field]

        return enriched


_engine_instance = None


def get_engine() -> ExplanationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ExplanationEngine()
    return _engine_instance


if __name__ == "__main__":
    engine = get_engine()
    for msg in [
        "NameError: name 'x' is not defined",
        "",
        "some completely unrelated nonsense text here",
    ]:
        print("INPUT:", repr(msg))
        print(engine.explain(msg))
        print()
