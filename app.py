"""Streamlit frontend.   streamlit run app.py"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import EMOTIONS, MODEL_PATH
from predictor import EmotionRecognizer, draw_results, open_camera

st.set_page_config(page_title="Emotion Recognition", page_icon="🙂", layout="wide")
st.title("🙂 Real-Time Facial Expression Recognition")
st.caption("Face detection (OpenCV) + custom CNN trained on FER-2013")


@st.cache_resource(show_spinner="Loading model...")
def get_recognizer():
    return EmotionRecognizer(MODEL_PATH)


try:
    recognizer = get_recognizer()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

mode = st.sidebar.radio("Input", ["Live webcam", "Snapshot (browser camera)", "Upload image"])


def show_results(bgr, results, image_slot, info_slot=None, chart_slot=None):
    image_slot.image(cv2.cvtColor(draw_results(bgr, results), cv2.COLOR_BGR2RGB),
                     use_container_width=True)
    if info_slot is None:
        return
    if not results:
        info_slot.info("No face detected")
        if chart_slot is not None:
            chart_slot.empty()
        return
    main = max(results, key=lambda r: r["box"][2] * r["box"][3])  # largest face
    info_slot.markdown(f"### Detected emotion: **{main['emotion'].upper()}** "
                       f"({main['confidence'] * 100:.1f}%)")
    if chart_slot is not None:
        chart_slot.bar_chart(pd.Series(main["probs"]).reindex(EMOTIONS))


def analyze_still(bgr):
    left, right = st.columns([3, 2])
    results = recognizer.analyze(bgr)
    image_slot, info_slot, chart_slot = left.empty(), right.empty(), right.empty()
    show_results(bgr, results, image_slot, info_slot, chart_slot)
    if len(results) > 1:
        st.write(f"{len(results)} faces found:")
        st.table(pd.DataFrame(
            [{"face": i + 1, "emotion": r["emotion"],
              "confidence": f"{r['confidence'] * 100:.1f}%"} for i, r in enumerate(results)]))


if mode == "Live webcam":
    st.write("Tick the box to start the webcam on this computer. Untick to stop.")
    run = st.checkbox("Start webcam")
    left, right = st.columns([3, 2])
    frame_slot, info_slot, chart_slot = left.empty(), right.empty(), right.empty()
    if run:
        cap = open_camera(0)
        if not cap.isOpened():
            st.error("Could not open the webcam. Close other apps using it, or use "
                     "'Snapshot' mode instead.")
        else:
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Lost webcam feed.")
                        break
                    frame = cv2.flip(frame, 1)
                    show_results(frame, recognizer.analyze(frame),
                                 frame_slot, info_slot, chart_slot)
            finally:
                cap.release()

elif mode == "Snapshot (browser camera)":
    shot = st.camera_input("Take a picture")
    if shot is not None:
        bgr = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        analyze_still(bgr)

else:
    up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if up is not None:
        bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read that image.")
        else:
            analyze_still(bgr)
