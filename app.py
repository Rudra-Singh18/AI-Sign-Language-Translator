from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

from utils import draw_annotation, extract_normalized_landmarks, guess_gesture

app = Flask(__name__)

MODEL_PATH = Path("models/gesture_classifier.h5")

LABELS = [
    "Open Palm",
    "Fist",
    "Thumbs Up",
    "Peace",
    "Pointing",
    "Unknown"
]

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

def load_model():

    if MODEL_PATH.exists():

        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(str(MODEL_PATH))

            print("Model Loaded Successfully")

            return model

        except Exception as e:

            print("Model Error:", e)

    return None

model = load_model()

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
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

            landmarks = extract_normalized_landmarks(
                hand_landmarks,
                width,
                height
            )

            if model is not None:

                logits = model.predict(
                    np.array([landmarks]),
                    verbose=0
                )[0]

                index = int(np.argmax(logits))

                confidence = float(logits[index])

                sign_text = f"{LABELS[index]} ({confidence:.2f})"

            else:

                sign_text = guess_gesture(landmarks)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        draw_annotation(frame, sign_text)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(debug=True)