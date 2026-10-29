# Yu-Gi-Oh Card Authenticator

## Overview
A machine learning-powered application to detect authentic Yu-Gi-Oh cards using computer vision and Roboflow's object detection model.

## Features
- Real-time camera capture
- Image preprocessing
- Card authenticity detection
- Confidence score reporting

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Roboflow API key as an environment variable: `export ROBOFLOW_API_KEY=...`
4. Run `python app.py` and open http://localhost:5000

## Requirements
- Python 3.8+
- Roboflow account
- Trained object detection model

## How It Works
1. Capture card image via camera
2. Preprocess and normalize image
3. Run inference with pre-trained model
4. Display authenticity result

## Limits
This was a fun local experiment. The dataset is my own cards, printed between 2005 and 2013,
so the model is biased towards those prints and is not expected to hold on recent cards.
