# AI Sign Language Translator

A starter project for a real-time sign language translator using a webcam. The application uses MediaPipe for hand detection and a simple gesture recognition pipeline to label common hand signs.

## Features

- Webcam capture for live hand gesture recognition
- MediaPipe Hands for landmark detection
- Heuristic fallback gesture recognition when a trained model is not available
- Optional TensorFlow model support for custom sign classification

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the application:

```powershell
python app.py
```

## Usage

- The app opens the default webcam and detects a single hand.
- It overlays the recognized sign label on the video stream.
- Press `Esc` to exit.

## Collecting training data

1. Run `python collect_data.py`.
2. Hold a gesture in front of the camera and press one of the number keys:
   - `1` — Open Palm
   - `2` — Fist
   - `3` — Thumbs Up
   - `4` — Peace
   - `5` — Pointing
3. Press `s` to save examples to `data/gesture_data.npz`.
4. Press `q` or `Esc` to quit.

## Training a gesture model

1. Run `python train.py`.
2. The script trains a simple gesture classifier from `data/gesture_data.npz`.
3. The trained model is saved to `models/gesture_classifier.h5`.

## Adding a trained model

- Place a TensorFlow Keras model file at `models/gesture_classifier.h5`
- The app will automatically load it and use the model for classification.

## Notes

- This project is a scaffold for building a full sign language translator.
- You can extend it by collecting a dataset, training a gesture classifier, and handling more signs.
