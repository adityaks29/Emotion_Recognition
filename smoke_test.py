"""End-to-end check of every module (uses synthetic data, 3 epochs, ~1-2 min).

    python smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
data, model_path = os.path.join(tmp, "data"), os.path.join(tmp, "m.keras")
env = dict(os.environ, EMOTION_MODEL=model_path, TF_CPP_MIN_LOG_LEVEL="2")


def run(*args):
    subprocess.run([sys.executable, *args], cwd=HERE, env=env, check=True)


print("1/5 model builds");   run("model.py")
print("2/5 sample data");    run("make_sample_data.py", data)
print("3/5 training (12 epochs)")
run("train.py", "--data-dir", data, "--out", model_path, "--epochs", "12")
assert os.path.isfile(model_path), "model file was not saved"

print("4/5 inference on a real photo + a blank image")
os.environ["EMOTION_MODEL"] = model_path
sys.path.insert(0, HERE)
from skimage import data as skdata
from predictor import EmotionRecognizer, draw_results

rec = EmotionRecognizer(model_path)
photo = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
res = rec.analyze(photo)
assert len(res) >= 1, "face not detected in test photo"
assert abs(sum(res[0]["probs"].values()) - 1) < 1e-4
assert draw_results(photo, res).shape == photo.shape
assert rec.analyze(np.zeros((240, 320, 3), np.uint8)) == []
print("   faces:", len(res), "->", res[0]["emotion"], round(res[0]["confidence"], 3))

print("5/5 Streamlit app")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=60).run()
assert not at.exception, at.exception
at.radio[0].set_value("Upload image").run()
assert not at.exception, at.exception

shutil.rmtree(tmp, ignore_errors=True)
print("\nALL CHECKS PASSED")
