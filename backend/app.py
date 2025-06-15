from flask import Flask, request, jsonify, render_template  # Import render_template
import os
import pandas as pd  # Required for input file processing
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Directory paths
MODEL_DIR = "model/model"
TEMP_DIR = "backend/temp"
MODEL_PATH = os.path.join(MODEL_DIR, "model.h5")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Check if model file exists and load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Root endpoint to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    print(f"Request received. Method: {request.method}")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Placeholder response
    return jsonify({'result': 'File received successfully'})

if __name__ == '__main__':
    app.run(debug=True)
