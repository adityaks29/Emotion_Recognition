# Real-Time Facial Expression Recognition

Webcam → face detection (OpenCV Haar cascade) → custom CNN (FER-2013) → emotion shown live in Streamlit.

Emotions: angry, disgust, fear, happy, neutral, sad, surprise.

## Project files
| File | Purpose |
|---|---|
| `config.py` | Paths, emotion names, colours |
| `model.py` | Custom CNN (4 conv blocks, BatchNorm, Dropout, augmentation inside) |
| `data.py` | Loads FER-2013 (CSV or folder format) |
| `train.py` | Trains, saves best model to `models/emotion_cnn.keras`, prints report |
| `predictor.py` | Face detection + emotion inference |
| `app.py` | Streamlit frontend (live webcam / snapshot / upload) |
| `realtime.py` | Simple OpenCV webcam window (alternative to Streamlit) |
| `smoke_test.py` | Automated end-to-end check (synthetic data) |

## 1. Setup (Python 3.9 – 3.12)
```bash
cd emotion_recognition
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

## 2. Check that everything works (optional, ~3 min)
```bash
python smoke_test.py          # should end with: ALL CHECKS PASSED
```
This uses a tiny synthetic dataset only to verify the code. It does not create the real model.

## 3. Get FER-2013 (free Kaggle account required)
Download **one** of these and put it in the `data/` folder:

* **Folder version** (easiest): https://www.kaggle.com/datasets/msambare/fer2013
  Unzip so you have `data/train/<emotion>/*.jpg` and `data/test/<emotion>/*.jpg`.
* **CSV version**: https://www.kaggle.com/datasets/deadskull7/fer2013
  Place the file at `data/fer2013.csv`.

## 4. Train
```bash
python train.py                 # default 40 epochs, early stopping
python train.py --epochs 60     # optional
```
CPU: about 30–60 min. Expect roughly **60–66 % validation accuracy**, typical for FER-2013 with a
small CNN (human agreement on this dataset is only ~65 %). The best model is saved to
`models/emotion_cnn.keras`; a report, confusion matrix and `emotion_cnn_history.png` are produced.

### Fast training (small subset of the real data)
```bash
python train.py --max-per-class 800 --epochs 25     # ~5-10 min on CPU, roughly 45-52 % accuracy
python train.py --max-per-class 2000 --epochs 30    # ~15-20 min, roughly 52-58 % accuracy
```
Uses a random subset of the real FER-2013 (still needs the dataset from step 3).

## 5. Run
```bash
streamlit run app.py      # opens http://localhost:8501
```
Choose **Live webcam** and tick *Start webcam*. If your browser/OS blocks the camera, use
**Snapshot** or **Upload image**. Alternative without Streamlit: `python realtime.py` (press `q` to quit).

## Troubleshooting
* *"Trained model not found"* → run step 4 first.
* *Webcam does not open* → close Zoom/Teams/other apps using it; try `python realtime.py --camera 1`.
* *Face not detected* → face the camera, good front lighting, at least ~1 foot away.
* Face detector file is bundled with OpenCV 4.x; if missing it is downloaded once to `models/`.
