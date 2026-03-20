from __future__ import annotations


TEXT_BLOCKS = {
    "project_overview": (
        "Pneumonia Detection & Synthetic Augmentation Explorer is an educational machine learning project "
        "that presents a medical imaging workflow from problem framing through evaluation, with a clear "
        "focus on interpretation as well as model development."
    ),
    "methodology_summary": (
        "The core task is binary image classification on chest X-rays, distinguishing NORMAL from PNEUMONIA. "
        "The project uses transfer learning as a practical baseline and compares that baseline with a stronger "
        "augmentation experiment to study whether broader training variation improves generalization."
    ),
    "interpretation_notes": (
        "Charts and model outputs should be read as decision-support style artifacts rather than standalone conclusions. "
        "The emphasis is on understanding class balance, model behavior, and tradeoffs across evaluation metrics."
    ),
    "limitations": (
        "This project is intended for research and demonstration. Dataset bias, label quality, limited sample diversity, "
        "and the gap between offline evaluation and clinical use all limit how far results should be interpreted."
    ),
    "synthetic_augmentation_explanation": (
        "In this first pass, the augmentation comparison represents stronger image transformations rather than truly synthetic "
        "clinical image generation. It is useful as an experiment in robustness, but it should not be presented as evidence of "
        "clinically realistic synthetic data."
    ),
    "recruiter_summary": (
        "This project highlights practical machine learning structure, thoughtful evaluation, "
        "clear communication, and the ability to turn technical work into an interactive experience."
    ),
}


PAGE_CONTENT = {
    "why_this_project": {
        "motivation": (
            "This project explores how computer vision can support pneumonia screening workflows while also showing how to present "
            "a medical imaging case study in a polished, easy-to-follow format."
        ),
        "importance": (
            "Pneumonia can progress quickly, and chest X-rays are a common part of the diagnostic pathway. That makes the task both "
            "clinically relevant and well suited to careful machine learning exploration."
        ),
        "portfolio": TEXT_BLOCKS["recruiter_summary"],
    },
    "data_and_context": {
        "summary": (
            "The dataset is organized into train, validation, and test splits with NORMAL and PNEUMONIA classes, creating a clear "
            "binary classification setup for exploration, modeling, and evaluation."
        ),
        "methodology": TEXT_BLOCKS["methodology_summary"],
        "limitations": TEXT_BLOCKS["limitations"],
    },
    "analytics_exploration": {
        "insight": (
            "Exploratory analytics help establish whether the dataset is balanced, whether image sizes are consistent, and whether "
            "the split structure looks trustworthy before any training decisions are made."
        ),
        "interpretation": TEXT_BLOCKS["interpretation_notes"],
    },
    "prediction_lab": {
        "notes": (
            "Prediction outputs are designed to show how a model might communicate confidence and visual explanation cues in a demo setting, "
            "while keeping the distinction clear between technical experimentation and real clinical use."
        ),
        "limitations": TEXT_BLOCKS["limitations"],
    },
    "model_performance": {
        "framing": (
            "Accuracy alone is not enough in a medical imaging workflow. Precision, recall, F1, ROC-AUC, and confusion patterns all help show "
            "how the model behaves under different types of error."
        ),
        "interpretation": TEXT_BLOCKS["interpretation_notes"],
    },
    "synthetic_augmentation": {
        "strategy": TEXT_BLOCKS["synthetic_augmentation_explanation"],
        "next_steps": (
            "A stronger next step would be to compare augmentation settings more systematically, review whether the holdout metrics improve, "
            "and inspect whether explanation maps remain plausible after heavier transformations."
        ),
    },
    "key_findings": {
        "highlights": [
            "The app combines project framing, analytics, prediction UX, and evaluation in one coherent flow.",
            "The structure makes it easier to compare a baseline transfer-learning model with an augmentation experiment.",
            "The presentation is designed to communicate technical depth without losing clarity for a broader audience.",
        ],
        "recommendation": (
            "The strongest version of this project will connect the polished front end to trained models, saved evaluation artifacts, "
            "and concise discussion of what the results do and do not support."
        ),
    },
}


def get_text_block(name: str) -> str:
    if name not in TEXT_BLOCKS:
        raise KeyError(f"Unknown text block `{name}`.")
    return TEXT_BLOCKS[name]


def get_page_section(page_name: str, section_name: str):
    if page_name not in PAGE_CONTENT:
        raise KeyError(f"Unknown page content group `{page_name}`.")
    page_sections = PAGE_CONTENT[page_name]
    if section_name not in page_sections:
        raise KeyError(f"Unknown section `{section_name}` for page `{page_name}`.")
    return page_sections[section_name]
