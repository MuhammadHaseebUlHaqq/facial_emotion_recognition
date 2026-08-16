# Facial Emotion Recognition

A deep learning project that classifies facial expressions into seven emotions
(**Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**) using a CNN trained on the
FER-2013 dataset, served through a Flask web app where you can upload a photo and see
the predicted emotion with per-class confidence scores.

---

## Contents

| Path | What it is |
|------|------------|
| `web_demo/app.py` | Flask web application (upload → face detection → prediction) |
| `web_demo/model.py` | CNN architecture definition and shared constants |
| `web_demo/train_model.py` | Training script (augmentation, class weights, callbacks) |
| `web_demo/download_fer2013.py` | Downloads FER-2013 from Hugging Face into `train/` and `test/` |
| `web_demo/emotion_model.keras` | Pre-trained model weights (~69 MB) |
| `web_demo/templates/`, `web_demo/static/` | Frontend (HTML, CSS, JS) |
| `Emotion_Recognition_DL_Project.ipynb` | Full experimentation notebook |
| `DL_Report_Moiz_Haseeb.pdf` | Project report |
| `DL_Presentation.pptx` | Project presentation |

---

## Model architecture

A "CNN + BatchNorm + Dropout" network for 48×48 grayscale images:

```
Block 1: Conv64  → BN → Conv64  → BN → MaxPool → Dropout(0.25)
Block 2: Conv128 → BN → Conv128 → BN → MaxPool → Dropout(0.25)
Block 3: Conv256 → BN → Conv256 → BN → MaxPool → Dropout(0.25)
Head   : Flatten → Dense512 → BN → Dropout(0.5)
                 → Dense256 → BN → Dropout(0.5)
                 → Dense7 (softmax)
```

Optimizer: Adam · Loss: categorical crossentropy

---

## Running the web demo

The trained model is committed to the repo, so you do **not** need the dataset or to
retrain anything just to run the demo.

### 1. Requirements

- Python 3.10–3.12 (developed on 3.12)
- ~2 GB free disk space for dependencies (TensorFlow is large)

### 2. Set up a virtual environment

**Windows (PowerShell)**

```powershell
cd web_demo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd web_demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Then open <http://localhost:5000> in your browser.

> **Note:** the first startup takes roughly 30–60 seconds. TensorFlow is slow to import
> and the 69 MB model has to be loaded, and Flask's debug reloader does it twice. This is
> normal — wait for the `Running on http://127.0.0.1:5000` line before opening the page.

Press `Ctrl+C` to stop the server.

### How prediction works

1. The uploaded image is converted to grayscale.
2. OpenCV's Haar Cascade detects faces; the largest face is cropped with 10% padding.
3. If no face is found, the whole image is used as a fallback.
4. The crop is resized to 48×48, normalised to `[0, 1]`, and passed to the model.
5. All seven class confidences are returned, sorted highest first.

---

## Training from scratch

### 1. Get the dataset

```bash
cd web_demo
python download_fer2013.py
```

This pulls FER-2013 from Hugging Face and writes it as
`<repo>/train/<emotion>/*.png` and `<repo>/test/<emotion>/*.png`.
The dataset is ~157 MB and is deliberately excluded from git via `.gitignore`.

If you already have FER-2013 laid out as `train/` and `test/` folders of per-emotion
subdirectories, you can skip this step.

### 2. Train

```bash
python train_model.py                      # expects train/ and test/ one level up
python train_model.py --data_dir /path/to/fer2013   # or point it somewhere else
```

Training details:

- 50 epochs max, batch size 64, 15% validation split, seed 42
- Augmentation: horizontal flip, ±15° rotation, 15% zoom, 10% shifts, brightness 0.8–1.2
- Balanced class weights to compensate for FER-2013's heavy class imbalance
- `EarlyStopping` (patience 10), `ReduceLROnPlateau` (patience 5), and `ModelCheckpoint`
  saving the best model to `emotion_model.keras`

Test accuracy is printed at the end of the run.

---

## Troubleshooting

**`Running on ...` never appears** — give it a full minute; TensorFlow import plus model
load is genuinely slow on first start.

**"TensorFlow GPU support is not available on native Windows"** — expected and harmless.
The app runs on CPU, which is plenty fast for single-image inference.

**"This is a development server"** — expected. `app.py` runs Flask's built-in server,
which is intended for local demos, not production deployment.

**Port 5000 already in use** — change the port on the last line of `app.py`, e.g.
`app.run(host="0.0.0.0", port=5001, debug=True)`. On macOS, AirPlay Receiver commonly
occupies port 5000.

---

## Authors

Moiz and Haseeb — Deep Learning course project.
