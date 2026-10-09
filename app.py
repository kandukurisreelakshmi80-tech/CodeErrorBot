"""
app.py
------
Streamlit UI for the Python Code Error Analyzer.
Run: streamlit run app.py
"""
from collections.abc import Mapping
import importlib.util
import os
import sys

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from code_analyzer import (  # noqa: E402
    MAX_SOURCE_LENGTH,
)
from explanation import get_engine  # noqa: E402
from program_explainer import explain_program  # noqa: E402

st.set_page_config(page_title="CodeErrorBot", page_icon="🧑‍💻", layout="centered")


@st.cache_resource
def load_engine():
    return get_engine()


def dependency_is_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _safe_text(value, fallback=""):
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _warning_explanation(category, message):
    explanations = {
        "UndefinedName": "This name is used but has not been defined. Check its spelling or define it before this line.",
        "UnusedImport": "This import is not used in the submitted code. Remove it if it is not needed.",
        "UnusedVariable": "This variable is assigned but not used. Check whether a later statement is missing.",
    }
    if category in explanations:
        return explanations[category]
    if "undefined name" in message.lower() or "not defined" in message.lower():
        return "A name may be used before it is defined. Check the spelling and where it is assigned."
    if message:
        return "Static analysis found a possible code-quality issue. Review the details and check whether this is intentional."
    return "Static analysis returned incomplete details, so more information is needed to explain this warning."


def _warning_location(warning):
    location = _safe_text(warning.get("location"))
    if location:
        return location
    line = warning.get("line")
    column = warning.get("column")
    if isinstance(line, int) and not isinstance(line, bool) and line > 0:
        if isinstance(column, int) and not isinstance(column, bool) and column > 0:
            return f"Line {line}, column {column}"
        return f"Line {line}"
    return "Location unavailable"


def show_warnings(result):
    if not isinstance(result, Mapping):
        return
    warnings = result.get("warnings", [])
    if warnings is None:
        warnings = []
    elif isinstance(warnings, Mapping) or isinstance(warnings, (str, bytes)):
        warnings = [warnings]
    elif not isinstance(warnings, (list, tuple)):
        st.warning("Static warning data was returned in an unexpected format and could not be fully displayed.")
        warnings = []

    if warnings:
        st.subheader("Static and possible logic warnings")
        for number, warning in enumerate(warnings, start=1):
            if not isinstance(warning, Mapping):
                st.warning(f"Warning {number} could not be displayed because its details were malformed.")
                continue
            category = _safe_text(warning.get("category"), "Static warning")
            message = _safe_text(warning.get("message"))
            explanation = _safe_text(warning.get("explanation")) or _warning_explanation(category, message)
            location = _warning_location(warning)
            st.warning(f"{location}: {explanation}")
            with st.expander(f"Static check details: {category} ({location})"):
                if message:
                    st.write(message)
                else:
                    st.write("No detailed message was provided.")
                code = _safe_text(warning.get("code"))
                if code:
                    st.text(code)
    dependency_notice = _safe_text(result.get("dependency_notice"))
    if dependency_notice:
        st.info(dependency_notice)


def show_import_checks(result):
    if not isinstance(result, Mapping):
        return
    imports = result.get("import_checks", [])
    if not isinstance(imports, (list, tuple)):
        return
    if not imports:
        return
    st.subheader("Import checks")
    for item in imports:
        if not isinstance(item, Mapping):
            continue
        module = _safe_text(item.get("module"), "Unknown module")
        line = item.get("line")
        location = f"Line {line}: " if isinstance(line, int) and not isinstance(line, bool) and line > 0 else ""
        kind = _safe_text(item.get("kind"))
        status = _safe_text(item.get("status"))
        if kind == "standard library":
            if status == "available":
                st.caption(f"{location}`{module}` is an available Python standard-library module.")
            else:
                st.warning(f"{location}`{module}` is part of the standard library but is unavailable in this Python installation; it is not a pip package.")
        elif status == "available":
            st.caption(f"{location}`{module}` is visible in this Python environment.")


def show_results(result):
    if not isinstance(result, Mapping):
        st.error("The analyzer returned an invalid result. Please submit the code again.")
        return

    show_warnings(result)
    show_import_checks(result)
    status = _safe_text(result.get("status"))

    if status == "empty":
        st.warning(_safe_text(result.get("message"), "Please enter Python source code."))
        return

    if status == "missing_dependencies":
        st.warning("Some imported modules are not available in the Python environment running this app.")
        st.write(_safe_text(result.get("message"), "Install the missing dependency or check whether it is a project-local module."))
        st.subheader("Missing dependencies")
        dependencies = result.get("missing_dependencies", [])
        if not isinstance(dependencies, (list, tuple)):
            dependencies = []
        for item in dependencies:
            if not isinstance(item, Mapping):
                st.warning("One missing-dependency entry was malformed and could not be displayed.")
                continue
            module = _safe_text(item.get("module"), "Unknown module")
            line = item.get("line")
            location = f" at line {line}" if isinstance(line, int) and not isinstance(line, bool) and line > 0 else ""
            st.markdown(f"**`{module}`{location}**")
            st.caption("This may be a third-party package or a project-local module that is not on the current import path.")
            install_command = _safe_text(item.get("install_command"))
            if install_command:
                st.text(install_command)
        st.info("These are suggestions only. CodeErrorBot does not install packages or execute submitted imports.")
        st.subheader("Submitted Python")
        source = result.get("source")
        if isinstance(source, str):
            st.text(source)
        return

    if status == "unsupported":
        st.warning("Some submitted Python features were not executed by the constrained analyzer.")
        st.write(_safe_text(result.get("message"), "Runtime analysis is unavailable for this Python feature."))
        st.info("Syntax and static checks can still run. Imports, file access, network access, and unsupported language features are never executed.")
        source = result.get("source")
        if isinstance(source, str) and source:
            st.text(source)
        return

    if status in {"timeout", "resource_limit", "execution_unavailable"}:
        st.warning(_safe_text(result.get("message"), "Runtime checks stopped; static analysis may still be available."))
        return

    if status == "error":
        issues = result.get("errors", [result])
        if isinstance(issues, Mapping):
            issues = [issues]
        elif not isinstance(issues, (list, tuple)):
            issues = [result]
        valid_issues = [issue for issue in issues if isinstance(issue, Mapping)]
        st.subheader("Confirmed analysis errors")
        st.markdown(f"**Errors found: {len(valid_issues)}**")
        if not valid_issues:
            st.warning("The result reported an error, but its details were incomplete. More information is needed to explain it reliably.")
        prevention_tips = {
            "NameError": "Define or import each name before using it, and check spelling and scope.",
            "UnboundLocalError": "Assign local variables before reading them inside a function.",
            "TypeError": "Check the types of values before combining them or passing them to a function.",
            "ValueError": "Validate the value before converting or processing it.",
            "IndexError": "Check a sequence's length before using an index.",
            "KeyError": "Check that a dictionary contains a key before requesting its value.",
            "ZeroDivisionError": "Check that a divisor is not zero before dividing.",
            "RecursionError": "Make recursive calls move toward a stopping condition, or use a loop.",
            "AssertionError": "Check the condition before relying on it being true.",
            "AttributeError": "Check the object's value and type before using an attribute or method.",
            "ModuleNotFoundError": "Check module spelling and install third-party packages in this same Python environment.",
            "ImportError": "Check imported names against the installed package version and avoid naming local files after packages.",
            "FileNotFoundError": "Check file paths and confirm required files exist in the program's working directory.",
            "SyntaxError": "Compile or lint changes before running the full program.",
            "IndentationError": "Use consistent indentation and keep statements inside the correct block.",
        }
        for number, issue in enumerate(valid_issues, start=1):
            error_type = _safe_text(issue.get("error_type"), "Python error")
            if error_type in {"SyntaxError", "IndentationError", "TabError"}:
                st.error("The submitted text is not valid Python syntax.")
            st.markdown(f"### Error {number}: `{error_type}`")
            line = issue.get("line")
            if isinstance(line, int) and not isinstance(line, bool) and line > 0:
                location = f"Line {line}"
                column = issue.get("column")
                if isinstance(column, int) and not isinstance(column, bool) and column > 0:
                    location += f", column {column}"
                st.markdown(f"**Location:** {location}")
            actual_message = _safe_text(issue.get("error_message"))
            if actual_message:
                st.markdown(f"**Observed Python message:** `{actual_message}`")
            code = _safe_text(issue.get("problematic_code"))
            if code:
                st.markdown("**Problematic line**")
                st.text(code)
            what_happened = _safe_text(issue.get("context_explanation")) or _safe_text(issue.get("explanation"))
            st.markdown("**Explanation**")
            st.write(what_happened or "The analyzer confirmed this error, but more context is needed to explain its exact cause.")
            st.markdown("**Hint**")
            st.write(_safe_text(issue.get("hint")) or prevention_tips.get(error_type, "Review the values and operation on this line."))
            install_command = _safe_text(issue.get("dependency_install"))
            if install_command:
                st.markdown(f"**Install the missing package in this environment:** `{install_command}`")
            predicted_type = _safe_text(issue.get("ml_predicted_type"))
            confidence = issue.get("ml_confidence")
            if predicted_type and isinstance(confidence, (int, float)):
                st.caption(
                    f"ML category: {predicted_type} "
                    f"({confidence * 100:.0f}% estimated confidence). "
                    "Runtime evidence is authoritative; this score is not a guarantee."
                )
            if number < len(valid_issues):
                st.divider()

        source = result.get("source")
        if isinstance(source, str):
            st.subheader("Submitted Python")
            st.text(source)
        with st.expander("Constrained runtime details"):
            traceback_text = _safe_text(result.get("traceback"), "No traceback was produced.")
            st.text(traceback_text)
            st.write(_safe_text(result.get("analysis_note"), "Runtime checks are limited to the supported Python subset."))
        return

    if status == "ok":
        st.success("No syntax or runtime errors detected in the supported analysis.")
        st.write(_safe_text(result.get("message"), "The supported analysis found no errors."))
        line_count = result.get("line_count", 0)
        variables = result.get("variables", [])
        if not isinstance(variables, (list, tuple, set)):
            variables = []
        st.caption(
            f"Analysis: {line_count if isinstance(line_count, int) else 0} lines, "
            f"{len(variables)} assigned variables, "
            f"{result.get('functions', 0)} functions, {result.get('imports', 0)} imports."
        )
        st.info("A clean constrained run does not prove logical correctness or guarantee behavior for other inputs.")
        return

    st.warning(_safe_text(result.get("message"), "The analyzer returned an incomplete result."))


st.markdown(
    '<h1 style="white-space: nowrap;"><strong style="font-size: 48px;">CodeErrorBot</strong> '
    '<span style="font-size: 30px; font-weight: normal;">- A Code Error Analyzer</span></h1>',
    unsafe_allow_html=True,
)

if st.button("Clear", type="secondary"):
    st.session_state["analysis_input"] = ""
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("program_explanation", None)
    st.rerun()

with st.form("python_analysis"):
    source = st.text_area(
        "Python source code",
        height=260,
        key="analysis_input",
        placeholder="numbers = [10, 20, 30]\nprint(numbers[5])",
    )
    submitted = st.form_submit_button("Analyze Python Code", type="primary", use_container_width=True)
    explain_submitted = st.form_submit_button("Explain Program", type="secondary", use_container_width=True)

st.caption(
    f"Input limit: {MAX_SOURCE_LENGTH:,} characters. Python imports are inspected but never executed by the constrained runtime checker."
)

try:
    engine = load_engine()
except Exception:
    engine = None
    st.error("The analysis engine could not load. Check the installed requirements and project files, then restart the app.")
if engine is not None and engine.predictor is None:
    st.warning("The saved ML classifier is unavailable or incompatible. Python AST and static analysis remain available; install the pinned requirements or retrain the model.")

if submitted:
    if not source.strip():
        st.session_state.pop("analysis_result", None)
        st.warning("Please enter Python source code.")
    elif engine is not None:
        with st.spinner("Analyzing Python source…"):
            result = engine.analyze_source(source)
            if isinstance(result, dict):
                result.setdefault("source", source)
            st.session_state["analysis_result"] = result

if explain_submitted:
    with st.spinner("Explaining the Python structure..."):
        explanation = explain_program(source)
        if isinstance(explanation, dict):
            explanation["source"] = source
        st.session_state["program_explanation"] = explanation

if engine is not None and st.session_state.get("analysis_result"):
    show_results(st.session_state["analysis_result"])

program_explanation = st.session_state.get("program_explanation")
if isinstance(program_explanation, Mapping) and program_explanation.get("source") == source:
    st.subheader("Program Explanation")
    if program_explanation.get("status") != "ok":
        st.warning(_safe_text(program_explanation.get("message"), "The program could not be explained."))
    else:
        for heading, paragraphs in program_explanation.get("sections", {}).items():
            st.markdown(f"### {heading}")
            if heading in {"PROGRAM OVERVIEW", "SIMPLE EXPLANATION"}:
                for paragraph in paragraphs:
                    st.write(paragraph)
            else:
                for paragraph in paragraphs:
                    st.markdown(f"- {paragraph}")

packages = [
    ("NumPy", "numpy", "numpy"),
    ("Pandas", "pandas", "pandas"),
    ("Matplotlib", "matplotlib", "matplotlib"),
    ("Requests", "requests", "requests"),
    ("Flask", "flask", "flask"),
    ("Streamlit", "streamlit", "streamlit"),
    ("Scikit-learn", "sklearn", "scikit-learn"),
    ("SciPy", "scipy", "scipy"),
    ("Seaborn", "seaborn", "seaborn"),
    ("Django", "django", "django"),
    ("Beautiful Soup", "bs4", "beautifulsoup4"),
    ("Pillow", "PIL", "Pillow"),
    ("OpenCV", "cv2", "opencv-python"),
    ("Pyflakes", "pyflakes", "pyflakes"),
    ("spaCy", "spacy", "spacy"),
    ("TensorFlow", "tensorflow", "tensorflow"),
    ("PyTorch", "torch", "torch"),
    ("Plotly", "plotly", "plotly"),
    ("OpenPyXL", "openpyxl", "openpyxl"),
    ("XGBoost", "xgboost", "xgboost"),
]
with st.expander("Dependency availability in this environment"):
    st.caption("Availability is checked in the Python environment running Streamlit. Imports in submitted code are not executed.")
    for display_name, module_name, package_name in packages:
        installed = dependency_is_available(module_name)
        if installed:
            st.write(f"{display_name}: installed")
        else:
            st.write(f"{display_name}: not installed. Install with `python -m pip install {package_name}`.")

st.divider()
st.caption("Python AST and compilation checks, static linting, bounded diagnostics for an allowlisted subset, and the existing traceback ML model when available.")
