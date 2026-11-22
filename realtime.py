"""Plain OpenCV webcam window.   python realtime.py   (press q to quit)"""
import argparse

import cv2

from predictor import EmotionRecognizer, draw_results, open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    rec = EmotionRecognizer()
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try --camera 1, and close other apps using it.")

    print("Running. Press 'q' in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            cv2.imshow("Emotion Recognition (q to quit)",
                       draw_results(frame, rec.analyze(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
