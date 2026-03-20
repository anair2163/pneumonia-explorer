# Data Notes

The raw chest X-ray dataset for this project is downloaded via `kagglehub` into `data/raw/`.

To populate that folder, users must have Kaggle access configured on their machine before running:

```bash
python scripts/download_data.py
```

Notes:

- `data/raw/` is the consistent local path the rest of the app can rely on
- raw data is excluded from git and should not be committed
- use this folder for the Kaggle dataset contents only, not for model artifacts
