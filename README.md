---
title: AI Sign Language Translator
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

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
