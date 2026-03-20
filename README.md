# Pneumonia Detection & Synthetic Augmentation Explorer

A polished Streamlit multipage app for presenting a pneumonia detection case study, comparing a baseline transfer-learning model with an augmentation experiment, and communicating the results in a recruiter-friendly format.

## First Run Checklist

- Install project dependencies with `pip install -r requirements.txt`
- Confirm the environment includes `streamlit`, `torch`, `torchvision`, and `kagglehub`
- Configure Kaggle or KaggleHub access on your machine
- Download the dataset into `data/raw/`
- Train a first baseline model, run evaluation, then launch the app

## Setup

```bash
pip install -r requirements.txt
```

The local app and ML scripts require `streamlit`, `torch`, `torchvision`, and `kagglehub`. Configure Kaggle or KaggleHub access on your machine before downloading data. The project uses the Kaggle dataset `paultimothymooney/chest-xray-pneumonia` and does not require the Kaggle CLI.

## Local Run Order

Run the project in this order:

```bash
pip install -r requirements.txt
python scripts/download_data.py
python src/train.py --epochs 1
python src/evaluate.py
streamlit run app.py
```

## Download the Dataset

```bash
python scripts/download_data.py
```

The raw dataset is downloaded into `data/raw/` and is excluded from git. Do not commit raw data to the repository.

## Train Models

Train the baseline model:

```bash
python src/train.py --experiment baseline
```

Train the augmentation experiment:

```bash
python src/train.py --experiment augmented
```

Model checkpoints and training history are saved under `models/`.

## Run Evaluation

```bash
python src/evaluate.py
```

This writes model summaries, per-image predictions, ROC data, and comparison files to `models/` so the Streamlit pages can read them directly.

## Launch the App

```bash
streamlit run app.py
```

The app is designed to run from the project root with relative, deployment-friendly paths.

## Deployment Notes

- Keep the repository root as the Streamlit working directory.
- Ensure `data/raw/` and any required model artifacts are present in the deployment environment.
- Update the GitHub and LinkedIn placeholder links before publishing.
- Large raw data files should be provisioned separately and not stored in Git.

## Structure

- `app.py` contains the home page and shared visual shell
- `pages/` contains the Streamlit multipage views
- `src/` holds reusable helpers for data loading, charts, inference, evaluation, and narrative content
- `scripts/download_data.py` downloads and syncs the Kaggle dataset into a consistent local `data/raw/` folder using `kagglehub`
- `models/` is reserved for trained checkpoints or serialized artifacts
- `assets/sample_images/` is a home for demo X-rays and screenshots
- `data/raw/` is the expected local location for raw Kaggle data
- `data/README.md` documents data sourcing and handling notes
