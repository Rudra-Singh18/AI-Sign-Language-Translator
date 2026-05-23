# Gesture Data

Use `collect_data.py` to record labeled hand landmarks from your webcam.

## Workflow

1. Run `python collect_data.py`.
2. Display your hand and press one of the number keys to label the current pose:
   - `1`: Open Palm
   - `2`: Fist
   - `3`: Thumbs Up
   - `4`: Peace
   - `5`: Pointing
3. Press `s` to save the dataset at any time.
4. Press `q` or `Esc` to exit.

The collected dataset is stored in `data/gesture_data.npz`, and `train.py` uses this file to train the classifier.
