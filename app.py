import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from utils import draw_annotation, extract_normalized_landmarks, guess_gesture

MODEL_PATH = Path("models/gesture_classifier.h5")
LABELS = ["Open Palm", "Fist", "Thumbs Up", "Peace", "Pointing", "Unknown"]


def load_model():
    if MODEL_PATH.exists():
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(str(MODEL_PATH))
            print(f"Loaded model from {MODEL_PATH}")
            return model
        except Exception as exc:
            print(f"Failed to load model: {exc}")
    print("No trained model found. Using heuristic gesture detection.")
    return None


def main():
    model = load_model()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open the webcam. Check your camera device.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            sign_text = "No hand detected"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                height, width, _ = frame.shape
                landmarks = extract_normalized_landmarks(hand_landmarks, width, height)

                if model is not None:
                    logits = model.predict(np.array([landmarks]), verbose=0)[0]
                    index = int(np.argmax(logits))
                    confidence = float(logits[index])
                    sign_text = f"Sign: {LABELS[index]} ({confidence:.2f})"
                else:
                    sign_text = f"Sign: {guess_gesture(landmarks)}"

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            draw_annotation(frame, sign_text)
            cv2.imshow("AI Sign Language Translator", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
