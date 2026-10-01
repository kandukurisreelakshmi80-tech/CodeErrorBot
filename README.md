# CodeErrorBot

CodeErrorBot is a Python-only Streamlit tool for checking Python source. It combines Python parsing and compilation, Pyflakes static diagnostics, constrained child-process runtime checks for an allowlisted subset, and the existing NLP classifier when its model artifacts load.

## Features

- Analyze a Python source submission only; there is no language selector or traceback-input mode in the UI.
- Show parser/compiler failures with their reported location and actual Python message.
- Report Pyflakes findings separately as static-analysis warnings. These are not presented as confirmed runtime failures.
- Flag constant/self-comparing conditions and statements after unconditional exits as possible logic or unreachable-code warnings, not confirmed bugs.
- Run supported source constructs in a separate child process after an explicit Analyze click. The child has a wall-clock timeout, an OS memory limit, bounded loops/arithmetic, a reduced built-in set, no submitted imports, and no file or network APIs.
- Preserve the saved TF-IDF classifier and explanation knowledge base as supplementary context for observed exceptions. Runtime evidence is authoritative.
- Explain errors without generating corrected code or exposing copy-code controls.
- Inspect all AST import statements, distinguish available standard-library modules from third-party/local modules, and report all missing third-party imports with mapped pip names. Comments and strings are ignored.
- Display the availability of common dependencies in the environment running Streamlit. Submitted imports are never executed; an unavailable package is named with a matching install command when possible.
- Use the existing classifier only for error-category prediction. No conversational AI API is configured, and no API key is needed.

## Project Audit

| File | Finding and impact | Change |
| --- | --- | --- |
| `app.py` | The UI exposed unused language choices and a separate traceback workflow; it had no reset or analysis spinner. | Reduced the workflow to Python source with reset, explicit submit, loading, separated warnings/errors, and dependency status. |
| `src/code_analyzer.py` | The source cap was 12,000 characters; code was passed as a Windows command-line argument; runtime had no OS memory limit and probed host paths for `open()` calls. | Raised the configurable default cap to 100,000 characters, moved source to stdin, bounded child resources, removed filesystem probing, and added Pyflakes checks. |
| `src/explanation.py` | A missing or incompatible saved model prevented the engine from starting, including for non-ML source analysis. | Keep AST/static analysis available if model loading fails; use ML enrichment only when it loads. |
| `requirements.txt` | No static linter was declared and scikit-learn was unpinned despite serialized model compatibility. | Pin Pyflakes 4.0.0 and scikit-learn 1.8.0 to verified/artifact versions. |
| `tests/test_source_analysis.py` | Existing tests did not cover larger files, timeout behavior, standard-library imports, package-name mappings, or model-load fallback. | Added these checks and verified the static warning path. |
| `venv/` (local environment) | The existing virtual environment refers to a Python installation that is no longer present. | Leave it untouched; create a fresh `.venv` using the commands below. |

The saved classifier was successfully loaded and its 27 existing held-out examples all matched their expected labels in this environment. The artifact emitted scikit-learn 1.8.0 versus installed 1.9.0 compatibility warnings; the requirements pin addresses this for fresh installs.

## Dependencies

Required packages from `requirements.txt`:

- `streamlit`, `nltk`, `pandas`, `numpy`, `joblib`
- `scikit-learn==1.8.0` for compatibility with the checked-in serialized model
- `pyflakes==4.0.0` for static source diagnostics

The app does not install packages automatically. Availability does not mean a user's imports are executed or that every package API is understood by the constrained runtime analyzer. For an uninstalled import, the result shows an install command (for example, `python -m pip install scikit-learn` for `sklearn`).

## Windows Setup and Run

From PowerShell:

```powershell
cd "D:\nlp project\CodeErrorBot"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If Python 3.12 is not installed, check installed interpreters with `py -0p` and use an available Python 3.11+ version. If PowerShell blocks activation, run `& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt` and `& ".\.venv\Scripts\python.exe" -m streamlit run app.py` instead.

The trained model artifacts are checked in under `model/`. If they are missing, run:

```powershell
python src/train_model.py
```

Source analysis remains available without model files; only the ML enrichment will be unavailable.

## Configuration

PowerShell environment variables (defaults shown):

```powershell
$env:CODEERRORBOT_MAX_SOURCE_CHARS = "100000"
$env:CODEERRORBOT_MAX_AST_NODES = "20000"
$env:CODEERRORBOT_EXECUTION_TIMEOUT = "2"
$env:CODEERRORBOT_MAX_MEMORY_MB = "128"
```

Input and AST limits are explicitly reported when exceeded; submissions are not truncated or chunked. Limits are clamped to safe upper bounds in the analyzer. Static checks run before constrained runtime diagnostics.

## Tests

```powershell
python tests/test_source_analysis.py
python tests/test_app_ui.py
python tests/test_cases.py
```

The source tests cover syntax and indentation errors, common runtime exception families, standard-library and third-party imports, mapped missing-package commands, relative imports, comments/strings, line locations, multiple Pyflakes warnings, possible logic/unreachable warnings, long source, model-load fallback, file-access rejection, and timeout handling. The UI suite covers analysis results and removed UI workflows. The ML suite contains 27 held-out traceback examples; that small set is not a real-world accuracy guarantee.

## Analysis and Security Limits

- Python AST parsing, compilation, and Pyflakes operate on the complete submitted source up to the configured character/AST limits. Pyflakes warnings are possible issues, not proof of runtime failure.
- Logic-risk warnings are heuristic and may be false positives or miss bugs.
- Runtime diagnostics support only a restricted AST/builtin subset. Imports, classes and other unsupported AST constructs, file access, dynamic code, and network access are not executed. Static checks and import availability checks still run for syntactically valid unsupported programs.
- Runtime checks use a separate child process. Windows uses a Job Object memory cap and active-process limit; POSIX uses address-space, CPU, and file-size limits. A wall-clock timeout applies on both. If the OS cannot apply the limits, runtime diagnostics are unavailable and static checks remain available.
- This is defense-in-depth for an educational local tool, not a general-purpose OS sandbox. The allowlist prevents submitted imports and file/network APIs, but it does not impose OS-level filesystem ACLs or network isolation. Do not expose this app as a public service for arbitrary untrusted submissions without an additional container/VM boundary, read-only filesystem, disabled egress, and service-level resource limits.
- Static analysis cannot reliably identify arbitrary logical mistakes or input-dependent runtime errors. The classifier is supplemental, trained on a small corpus, and its confidence is an estimate. Serialized scikit-learn artifacts should be loaded with the pinned version.
- Some third-party APIs and Python syntax constructs are not interpreted by the constrained runtime analyzer. Their source is parsed and linted where supported, but runtime behavior is not verified.

## Deployment

For Streamlit Community Cloud, push the project with `app.py`, `src/`, `dataset/`, `model/`, and `requirements.txt` included. Create an app for the repository/branch and set the main file to `app.py`; the platform installs `requirements.txt`. Confirm the model artifacts are present in the deployed revision. Do not add secrets to the repository.

Because submitted code is untrusted and this application does not provide OS-level filesystem/network isolation, deploy only to a trusted audience unless you add a separately enforced container or VM sandbox. For container deployment, run as a non-root user with a read-only application filesystem, no network egress for the analyzer, capped memory/CPU/PIDs, and no mounted host credentials or secrets.