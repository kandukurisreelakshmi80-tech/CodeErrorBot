# CodeErrorBot Project Explanation

## 1. What the project does

CodeErrorBot is a Streamlit application for analyzing Python source code. A user pastes Python code and chooses **Analyze Python Code**. The project checks whether the code is valid Python, reports static-analysis warnings, checks imported module availability, and runs a limited set of supported code in a separate restricted process to find runtime errors. It then shows the error location, the relevant source line, a plain-language explanation, and a hint.

The machine-learning model is supplementary. It classifies an error message into an error category after an error has been observed; it does not inspect Python source to discover errors by itself. Runtime evidence is treated as authoritative. CodeErrorBot does not generate corrected code.

## 2. Complete program flow

1. `app.py` starts the Streamlit page and loads the analysis engine. The engine is cached so the saved model is not repeatedly loaded during normal app interactions.
2. The user enters Python source and submits the form. Empty input is handled by the app and analyzer without further analysis.
3. `ExplanationEngine.analyze_source()` in `src/explanation.py` passes the source to `analyze_source()` in `src/code_analyzer.py`.
4. The analyzer checks the input-size limit, parses the text into a Python AST (Abstract Syntax Tree), and compiles that tree. A parsing or compilation failure, such as `SyntaxError` or `IndentationError`, is returned with its line and column when available.
5. For valid syntax, the analyzer walks the AST to inspect imports and check whether modules are available in the active Python environment. It does not import or run the submitted imports. It also runs Pyflakes and CodeErrorBot's own logic-warning checks.
6. If an imported third-party or local module is not found, the analyzer reports it and stops before runtime analysis. It may show a suggested pip package name; it does not install anything.
7. The analyzer rejects file access and Python syntax or operations outside its allowlist. Static checks and import checks can still be reported for unsupported source.
8. If the source passes those checks, it is sent to a separate child Python process. The child receives a restricted set of built-ins, bounded operations and loops, and operating-system resource limits. It captures output and reports exceptions instead of running arbitrary submitted imports or file/network operations.
9. If the child reports an exception, the analyzer records the type, message, source line, problematic code, and traceback. It can rerun analysis with a failing statement skipped to look for other independent errors, up to its retry limit.
10. `explanation.py` adds a beginner-friendly explanation, cause, and hint from `dataset/explanations.json` or built-in explanations. It adds a little source-specific context for some `NameError` and `IndexError` cases. If the saved classifier is available, it also attaches its predicted error type and confidence as supplemental information.
11. `app.py` displays errors, static warnings, import checks, or the result status. Static warnings are shown separately from confirmed runtime errors.

In short:

`Python source -> AST parse/compile -> import and static checks -> allowlist checks -> bounded runtime check -> explanation and optional ML label -> Streamlit result`

## 3. AST, static checks, and error detection

### AST and syntax

Python's built-in `ast` module converts source text into a tree of nodes representing statements and expressions. CodeErrorBot uses this tree to find imports, count or inspect constructs, identify certain suspicious conditions, enforce the supported-syntax allowlist, and transform a few operations to bounded helper functions. Python parsing and compilation detect malformed syntax before the runtime child process is started.

### Static analysis

Pyflakes checks syntactically valid source without running it. It can report possible problems such as undefined names and unused imports or variables. CodeErrorBot also has AST-based heuristics for constant `if`/`while` conditions, comparisons of a name with itself, and statements after an unconditional `return`, `break`, or `continue`. These are warnings: they may indicate a mistake, but they are not proof that the program will fail.

### Runtime error detection

For supported constructs, the child process executes the submitted program in a restricted namespace. It can detect exceptions raised by the actual operations, including errors such as `NameError`, `TypeError`, `IndexError`, `KeyError`, `ZeroDivisionError`, and `AttributeError`. The analyzer uses the traceback to find the submitted-code line and pairs it with the original source line.

The runtime pass is deliberately limited. Imports are inspected but not executed; file access is rejected; unsupported AST features, unbounded `while` loops, and certain risky operations are not run. There are time, memory, and operation limits. Therefore, an `ok` result means no error was found during this supported, constrained run; it does not prove the program is correct for every input or that its logic is right.

## 4. NLP and machine-learning methods actually present

The NLP pipeline is for **error-message text**, not for parsing submitted source code.

### Text preprocessing

`src/preprocessing.py` lowercases an error message, removes punctuation that is not useful for classification while retaining letters, digits, underscores, dots, and spaces, tokenizes it, removes common stop words while preserving useful phrases such as "not defined" and "no module," and lemmatizes words. It uses NLTK tokenization, English stop words, and WordNet lemmatization when available. If NLTK or its data is unavailable, it uses simple built-in tokenization, a local stop-word list, and conservative fallback lemmatization. The same `preprocess()` function is used during training and prediction so that the text is prepared consistently.

### TF-IDF and classification

`src/train_model.py` turns the processed messages into numeric TF-IDF features. The vectorizer uses single words and two-word phrases (unigrams and bigrams), so phrases such as "not defined" can be useful signals. TF-IDF gives more weight to terms that help distinguish messages and less weight to terms common across messages. This is a word/phrase-based representation, not a neural embedding or a system that understands arbitrary code semantics.

The training script compares three supervised classifiers:

- Logistic Regression
- Multinomial Naive Bayes
- A calibrated Linear Support Vector Machine (Linear SVM)

It uses a stratified 80/20 train/test split with a fixed random seed, prints accuracy, weighted F1, and a classification report, and saves the model with the best **weighted F1** on that split. The exact selected classifier can depend on the measured training run; the code does not hard-code one winner.

At prediction time, `src/predictor.py` preprocesses an error message, transforms it with the saved vectorizer, and asks the saved classifier for an error-type label. If the classifier provides probabilities, the largest probability is used as the confidence estimate. A confidence below 0.45 is treated as uncertain by the predictor.

### Similarity

The predictor also contains a cosine-similarity fallback. It compares a low-confidence error message's TF-IDF vector with saved preprocessed training messages and can return nearby known examples when similarity is at least 0.20. This fallback is used by `ExplanationEngine.explain()` for a supplied error message when its prediction is uncertain. The Streamlit source-analysis flow calls `analyze_source()` instead: for a confirmed runtime error it attaches the classifier's category and confidence, but does not use the similarity fallback to detect the source error.

There is no generative AI, chat model, transformer, or semantic code-understanding model in this project.

## 5. Dataset and saved model files

### Dataset

`dataset/errors.csv` contains example error messages and their error-type labels, along with explanatory columns. The training loader in `src/train_model.py` specifically uses `error_message` as the input text and `error_type` as the target label. It drops rows missing either of those fields and duplicate error messages. The other CSV columns are not training features.

`dataset/generate_dataset.py` creates the CSV examples for 12 common Python error categories. `dataset/explanations.json` is a separate knowledge base of human-written explanations, causes, suggestions, and examples. The source-analysis explanation path uses the explanation and cause text, plus hints defined in `src/explanation.py`; it does not train the classifier from the JSON file.

### Model artifacts

The training script saves these files in `model/`:

- `error_classifier.pkl`: the selected trained classifier. `predictor.py` loads it to predict an error category.
- `tfidf_vectorizer.pkl`: the fitted TF-IDF vocabulary and transformation settings. `predictor.py` needs the matching vectorizer to turn new messages into the same feature format used in training.
- `similarity_corpus.pkl`: the preprocessed training messages, labels, and raw messages. The predictor uses this optional file for cosine-similarity matches.
- `model_name.pkl`: the selected classifier's name, saved as metadata by the training script. The current prediction code does not load this metadata file.

These files are loaded with Joblib. If the classifier or vectorizer is missing or incompatible, the source-analysis engine can still perform AST, static, and constrained runtime analysis; the ML enrichment is unavailable.

## 6. Python libraries used

### Third-party packages

| Library | What it is used for |
| --- | --- |
| Streamlit | Builds the browser-based input form, buttons, status messages, warnings, and error results in `app.py`. |
| scikit-learn | Provides TF-IDF, the training classifiers, train/test splitting, evaluation metrics, and cosine similarity. |
| NLTK | Optional tokenization, English stop words, and WordNet lemmatization in preprocessing. Built-in fallbacks are used if it is unavailable. |
| Joblib | Saves and loads the trained classifier, vectorizer, and similarity corpus. |
| NumPy | Finds the highest predicted probability and sorts similarity scores in the predictor. |
| Pandas | Reads and prepares the CSV dataset for model training and exploratory data analysis. |
| Pyflakes | Reports static source-code warnings. If it is unavailable, the analyzer continues with its own AST warnings and shows a dependency notice. |

These are the third-party packages listed in `requirements.txt`. NLTK is optional at runtime in the preprocessing implementation, although it is listed as a project dependency.

### Python standard library

The code also uses standard-library modules; these are included with Python rather than installed with pip. The most important ones are `ast` for parsing and inspecting Python syntax, `subprocess` for the bounded child process, `traceback` for error locations, `json` for communication with that process, `os` and `sys` for paths and environment handling, `re` for text/context matching, and `builtins`, `contextlib`, `io`, `operator`, `math`, `ctypes`, and (where supported) `resource` for restricted execution and resource controls. The dataset generator uses `csv` to write its output.

## 7. Important project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface; accepts Python source, calls the explanation engine, and renders analysis results. |
| `src/code_analyzer.py` | Parses and compiles source, checks imports and static warnings, enforces runtime restrictions, runs supported source in a child process, and collects errors. |
| `src/explanation.py` | Connects the analyzer to the explanation knowledge base and optional classifier; adds explanations, causes, hints, and some source-specific context. |
| `src/preprocessing.py` | Provides the shared error-message text-cleaning and NLP preprocessing pipeline for both training and prediction. |
| `src/predictor.py` | Loads saved model artifacts, classifies error-message text, estimates confidence, and implements the cosine-similarity fallback. |
| `src/train_model.py` | Loads the CSV, preprocesses and vectorizes its messages, compares classifiers, evaluates them, and saves the selected artifacts. |
| `src/eda.py` | Prints a summary of the dataset, categories, missing values, duplicates, and category distribution. It is an analysis utility, not part of the Streamlit request flow. |
| `dataset/errors.csv` | Labeled example error messages used to train and test the classifiers. |
| `dataset/explanations.json` | Human-written explanation knowledge base used to describe detected error categories. |
| `dataset/generate_dataset.py` | Generates the labeled CSV examples. |
| `tests/` | Contains source-analysis, UI, and saved-classifier test scripts. |
| `model/` | Contains the saved classifier, TF-IDF vectorizer, and similarity data used by prediction. |
| `requirements.txt` | Lists the project's third-party Python dependencies. |

## 8. Example: a list index error

Suppose the user submits:

```python
numbers = [10, 20, 30]
print(numbers[5])
```

1. Python parses and compiles the source successfully, so there is no syntax error.
2. CodeErrorBot checks the AST and static warnings. There are no imports that need availability checks.
3. The code uses supported operations, so the analyzer runs it in the restricted child process.
4. The second line asks for list index `5`. A three-item list has valid indexes `0`, `1`, and `2`, so Python raises `IndexError: list index out of range`.
5. The analyzer reports the error type, line 2, and `print(numbers[5])`. The explanation engine uses the `IndexError` knowledge-base entry and recognizes the list and index in the source to give context such as: the list contains 3 elements, valid indexes are 0 through 2, and index 5 is outside that range.
6. If the saved classifier is available, it also classifies the observed error message and shows its estimated ML category and confidence. The actual runtime error remains the primary result.

The app explains the problem; it does not rewrite the line or provide corrected code.

## 9. Short presentation / viva explanation

> CodeErrorBot is a Python code analyzer with a Streamlit interface. The user pastes Python source and submits it for analysis. First, the program parses the source into an Abstract Syntax Tree and checks that it compiles. It then checks imports and runs static analysis with Pyflakes, along with a few AST-based checks for suspicious conditions and unreachable statements. These warnings are kept separate from confirmed errors.
>
> If the source uses supported Python features, CodeErrorBot runs it in a separate process with restricted built-ins and time and memory limits. When an exception occurs, the analyzer reports its type, line, and source code. The explanation engine adds a beginner-friendly description and hint from a JSON knowledge base, and may add details based on the source.
>
> The project also has an NLP and machine-learning component for classifying error messages. It preprocesses text using NLTK when available, converts messages into unigram and bigram TF-IDF features, and compares Logistic Regression, Multinomial Naive Bayes, and a calibrated Linear SVM during training. The best model by weighted F1 is saved and loaded to provide a supplementary error-category prediction. In short, static and constrained runtime analysis find source problems, while the ML model helps label observed error messages.
