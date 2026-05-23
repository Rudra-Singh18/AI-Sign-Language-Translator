import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from utils import extract_normalized_landmarks, draw_annotation

LABELS = ["Open Palm", "Fist", "Thumbs Up", "Peace", "Pointing"]
DATA_PATH = Path("data")
DATA_FILE = DATA_PATH / "gesture_data.npz"


def load_existing_data():
    if DATA_FILE.exists():
        with np.load(DATA_FILE) as data:
            return list(data["X"]), list(data["y"])
    return [], []


def save_data(X, y):
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_FILE, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int32))
    print(f"Saved {len(X)} examples to {DATA_FILE}")


def format_counts(counts):
    return " | ".join(f"{label}: {count}" for label, count in zip(LABELS, counts))


def main():
    X, y = load_existing_data()
    counts = [0] * len(LABELS)
    for label in y:
        counts[label] += 1

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open the webcam.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            landmarks = None
            if results.multi_hand_landmarks:
                landmarks = extract_normalized_landmarks(
                    results.multi_hand_landmarks[0], frame.shape[1], frame.shape[0]
                )
                mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            draw_annotation(frame, "Press 1-5 to record, s to save, q/Esc to quit")
            draw_annotation(frame, format_counts(counts), y_offset=80)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("s"):
                save_data(X, y)
            if landmarks is not None and key in [ord(str(i)) for i in range(1, len(LABELS) + 1)]:
                index = int(chr(key)) - 1
                X.append(landmarks)
                y.append(index)
                counts[index] += 1
                print(f"Recorded {LABELS[index]} ({counts[index]})")

    cap.release()
    cv2.destroyAllWindows()
    if X:
        save_data(X, y)


if __name__ == "__main__":
    main()
