from __future__ import annotations

import contextlib
import importlib
import runpy
import sys
from io import StringIO
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluate import MODEL_REGISTRY
from src.load_data import detect_dataset_root, to_project_relative_path
from src.train import MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULES_TO_IMPORT = [
    "src.load_data",
    "src.train",
    "src.evaluate",
    "src.infer",
    "src.explain",
    "src.charts",
    "src.text_sections",
]
PAGE_FILES_TO_CHECK = [
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "pages" / "1_Why_This_Project.py",
    PROJECT_ROOT / "pages" / "2_Data_and_Context.py",
    PROJECT_ROOT / "pages" / "3_Analytics_Exploration.py",
    PROJECT_ROOT / "pages" / "4_Prediction_Lab.py",
    PROJECT_ROOT / "pages" / "5_Model_Performance.py",
    PROJECT_ROOT / "pages" / "6_Synthetic_Augmentation.py",
    PROJECT_ROOT / "pages" / "7_Key_Findings.py",
]


def check_dataset_ready() -> tuple[bool, str]:
    data_root = PROJECT_ROOT / "data" / "raw"
    try:
        dataset_root = detect_dataset_root(data_root, verbose=False, require_complete=True)
    except FileNotFoundError as exc:
        return False, str(exc)

    return True, f"Detected dataset root at `{to_project_relative_path(dataset_root)}`."


def check_baseline_model_ready() -> tuple[bool, str]:
    if MODEL_PATH.exists():
        return True, f"Found baseline model at `{to_project_relative_path(MODEL_PATH)}`."
    return False, (
        f"Missing baseline model file `{to_project_relative_path(MODEL_PATH)}`. "
        "Run `python src/train.py --epochs 1`."
    )


def check_evaluation_artifacts_ready() -> tuple[bool, str]:
    baseline_entry = MODEL_REGISTRY["baseline"]
    required_paths = [
        Path(baseline_entry["summary_path"]),
        Path(baseline_entry["predictions_path"]),
        Path(baseline_entry["sample_predictions_path"]),
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    if not missing_paths:
        available_parts = ", ".join(to_project_relative_path(path) for path in required_paths)
        return True, f"Found baseline evaluation artifacts: {available_parts}."

    missing_text = ", ".join(to_project_relative_path(path) for path in missing_paths)
    return False, (
        f"Missing evaluation artifact(s): {missing_text}. "
        "Run `python src/evaluate.py` after training the baseline model."
    )


def check_module_imports() -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    for module_name in MODULES_TO_IMPORT:
        try:
            with _capture_noisy_output():
                importlib.import_module(module_name)
        except Exception as exc:
            ok = False
            messages.append(f"[module] {module_name}: FAILED ({type(exc).__name__}: {exc})")
        else:
            messages.append(f"[module] {module_name}: OK")

    return ok, messages


def check_page_files() -> tuple[bool, list[str]]:
    messages: list[str] = []
    ok = True

    for page_path in PAGE_FILES_TO_CHECK:
        if not page_path.exists():
            ok = False
            messages.append(f"[page] {page_path.name}: MISSING")
            continue

        try:
            with _capture_noisy_output():
                runpy.run_path(str(page_path), run_name="__main__")
        except Exception as exc:
            ok = False
            messages.append(f"[page] {page_path.name}: FAILED ({type(exc).__name__}: {exc})")
        else:
            messages.append(f"[page] {page_path.name}: OK")

    return ok, messages


def print_section(title: str, lines: list[str]) -> None:
    print(f"\n{title}")
    for line in lines:
        print(f"  - {line}")


def format_yes_no(value: bool) -> str:
    return "yes" if value else "no"


@contextlib.contextmanager
def _capture_noisy_output():
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        yield


def main() -> None:
    print("Pneumonia Explorer Smoke Test")
    print(f"Project root: {PROJECT_ROOT}")

    dataset_ready, dataset_message = check_dataset_ready()
    baseline_ready, baseline_message = check_baseline_model_ready()
    evaluation_ready, evaluation_message = check_evaluation_artifacts_ready()
    modules_ok, module_messages = check_module_imports()
    pages_ok, page_messages = check_page_files()

    print_section("Environment Checks", [dataset_message, baseline_message, evaluation_message])
    print_section("Module Imports", module_messages)
    print_section("Page Imports", page_messages)

    app_runnable = modules_ok and pages_ok

    print("\nReadiness Summary")
    print(f"  - Dataset ready: {format_yes_no(dataset_ready)}")
    print(f"  - Baseline model ready: {format_yes_no(baseline_ready)}")
    print(f"  - Evaluation artifacts ready: {format_yes_no(evaluation_ready)}")
    print(f"  - App likely runnable: {format_yes_no(app_runnable)}")

    if not app_runnable:
        print("\nOne or more imports failed. Review the messages above before launch.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[smoke_test] Smoke test failed unexpectedly.")
        print(f"[smoke_test] {type(exc).__name__}: {exc}")
        raise SystemExit(1)
