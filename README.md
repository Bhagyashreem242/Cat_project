# Cat Emotion Detection

Classifies cat audio (WAV/MP3) into: **angry**, **fighting**, **happy**, **sad**

## Project Structure
```
cat_project/
├── backend/
│   ├── app.py               # Flask web app
│   ├── model.h5             # Trained model (required)
│   ├── templates/index.html
│   └── static/
│       ├── style.css
│       └── uploads/         # Uploaded audio saved here
├── cat_dataset/             # Raw audio files per emotion
├── spectrogram_dataset/     # Generated spectrograms per emotion
├── convert_to_spectrogram.py
├── train_model.py
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Workflow

### Step 1 — Convert audio to spectrograms (only if retraining)
```bash
cd cat_project
python convert_to_spectrogram.py
```

### Step 2 — Train the model (only if retraining)
```bash
cd cat_project
python train_model.py
# Saves model to backend/model.h5 automatically
```

### Step 3 — Run the web app
```bash
cd cat_project/backend
python app.py
```
Then open: http://127.0.0.1:5000

Upload a `.wav` or `.mp3` cat sound and get the predicted emotion + confidence.

## Bugs Fixed
| File | Bug | Fix |
|------|-----|-----|
| `app.py` | `n_mels=128` vs training's `n_mels=256` | Changed to `256` |
| `app.py` | `fmax=8000` vs training's `fmax=10000` | Changed to `10000` |
| `app.py` | `cmap='magma'` vs training's `cmap='inferno'` | Changed to `'inferno'` |
| `app.py` | `figsize=(3,3)` vs training's `figsize=(4,4)` | Changed to `(4,4)` |
| `app.py` | No silence trim/normalize in inference | Added `trim` + `normalize` |
| `app.py` | `model.h5` loaded from CWD (fragile) | Load relative to `__file__` |
| `app.py` | No error handling on prediction | Added try/except + error display |
| `train_model.py` | `ModelCheckpoint` path `'backend/model.h5'` breaks if run from wrong dir | Use absolute path via `__file__` |
