import cv2
import mediapipe as mp
import numpy as np
import os

gesture = input("Enter alphabet (A-Z): ").upper()

save_dir = f"data/{gesture}"

os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:

                landmarks.extend([lm.x, lm.y, lm.z])

            cv2.putText(
                frame,
                f"Hand Detected | Samples: {count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    else:

        cv2.putText(
            frame,
            "No Hand Detected",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and results.multi_hand_landmarks:

        np.save(
            f"{save_dir}/{count}.npy",
            np.array(landmarks)
        )

        count += 1

        print(f"Saved sample {count}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()