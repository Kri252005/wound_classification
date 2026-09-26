# WoundScope training

This training script is reusable: it reads class names from `train/` in alphabetical order and requires the same folders in `val/` and `test/`.

## Add `Other`

Add `Other/` to each split and place non-wound images there. Keep all classes in each split, do not duplicate images across splits, and do not put near-duplicates from one image source into different splits.

## Run

```powershell
cd C:\Users\kri20\Documents\Codex\2026-09-01\build-x20
py -m venv training\.venv
.\training\.venv\Scripts\Activate.ps1
pip install -r training\requirements.txt
python training\train_wound_classifier.py --data-root C:\path\to\Wound_dataset_split --output-dir training_output
```

Copy the resulting `model.weights.h5`, `config.json`, and `metadata.json` to `backend/model/`. A seven-class artifact also requires the backend class mapping update before it is deployed.
