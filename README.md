# Chest X-Ray Classification and Augmentation Explorer

A Streamlit app comparing two ResNet-18 transfer-learning runs on a public pediatric chest X-ray dataset. The app presents class balance, threshold behavior, confusion matrices and a side-by-side training augmentation experiment.

[Evaluation metrics](models/evaluation_comparison.csv) · [Source dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Results on the supplied 624-image test split

| Run | ROC-AUC | F1 | Pneumonia recall | Normal specificity |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.966 | 0.883 | 99.7% | 56.4% |
| Stronger augmentation | 0.958 | 0.881 | 99.5% | 56.0% |

The baseline classified 389 of 390 pneumonia images correctly and flagged 102 of 234 normal images as pneumonia. That tradeoff is more informative than recall alone. The stronger augmentation run scored slightly lower on the saved metrics. Each experiment ran for one epoch with one seed, so the difference is descriptive. The comparison uses image transformations and the same supplied test split for both runs.

This is a dataset-specific educational evaluation. The repository does not establish external, patient-grouped or clinical validation. Displayed model scores are uncalibrated.

## Run locally

Inspect the saved evaluation results through the app:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

To regenerate the results and enable live inference, configure KaggleHub access and run from the repository root:

```bash
python scripts/download_data.py
python src/train.py --experiment baseline --epochs 1 --seed 42
python src/train.py --experiment augmented --epochs 1 --seed 42
python src/evaluate.py
streamlit run app.py
```

Training writes local checkpoints under `models/`. The repository includes evaluation summaries and prediction tables. Raw images and checkpoints are provisioned locally.

Data source: Kermany, Goldbaum, Cai et al., [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://doi.org/10.1016/j.cell.2018.02.010), *Cell*, 2018.
