# app.py
from flask import Flask, render_template, request, jsonify, Response
import cv2
from roboflow import Roboflow
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Initialize Roboflow
rf = Roboflow(api_key="rdYBlkQwzVCiBbpoIcV3")
project = rf.workspace("davide-iotti").project("fake-yu-gi-oh-card-detector")
model = project.version(1).model

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Draw guide rectangle
            height, width = frame.shape[:2]
            guide_margin = 100
            cv2.rectangle(frame, 
                         (guide_margin, guide_margin), 
                         (width - guide_margin, height - guide_margin), 
                         (0, 255, 0), 2)
            
            # Add text
            cv2.putText(frame, 'Position card here', (width//2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture image'}), 400
    
    # Save captured frame
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(UPLOAD_FOLDER, f'capture_{timestamp}.jpg')
    cv2.imwrite(filepath, frame)
    
    try:
        # Get prediction
        prediction = model.predict(filepath).json()
        predictions = prediction['predictions'][0]['predictions']
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"upload_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        # Get prediction
        prediction = model.predict(filepath).json()
        predictions = prediction['predictions'][0]['predictions']
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)