import cv2
import numpy as np


def extract_normalized_landmarks(hand_landmarks, image_width, image_height):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks, dtype=np.float32)


def count_extended_fingers(landmarks):
    points = [(landmarks[i * 3], landmarks[i * 3 + 1]) for i in range(21)]
    extended = 0

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip_y = points[tip_idx][1]
        pip_y = points[pip_idx][1]
        if tip_y < pip_y:
            extended += 1

    thumb_tip_x = points[4][0]
    thumb_ip_x = points[3][0]
    if thumb_tip_x < thumb_ip_x:
        extended += 1

    return extended


def guess_gesture(landmarks):
    points = [(landmarks[i * 3], landmarks[i * 3 + 1]) for i in range(21)]
    extended = count_extended_fingers(landmarks)

    if extended == 5:
        return "Open Palm"
    if extended == 0:
        return "Fist"

    thumb_extended = points[4][0] < points[3][0]
    index_extended = points[8][1] < points[6][1]
    middle_extended = points[12][1] < points[10][1]
    ring_extended = points[16][1] < points[14][1]
    pinky_extended = points[20][1] < points[18][1]

    if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Thumbs Up"
    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "Peace"
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "Pointing"

    return "Unknown"


def draw_annotation(frame, text, y_offset=40):
    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 240, 240), 2, cv2.LINE_AA)
