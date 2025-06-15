from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Directory paths
MODEL_PATH = "model/fingerprint_blood_group_model.h5"
TEMP_DIR = "backend/temp"

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Define the label mapping
class_labels = ["A+", "B+", "AB+", "O+", "A-", "B-", "AB-", "O-"]

# Root endpoint to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Image preprocessing
def preprocess_image(img_path):
    IMG_HEIGHT, IMG_WIDTH = 64, 64  # Match training size
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_resized = cv2.resize(image_rgb, (IMG_HEIGHT, IMG_WIDTH))
    image_normalized = image_resized / 255.0  # Normalize pixel values
    image_expanded = np.expand_dims(image_normalized, axis=0)  # Add batch dimension
    return image_expanded

# Prediction function
def predict_blood_group(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get class index
    confidence = np.max(prediction) * 100  # Confidence score
    blood_group = class_labels[predicted_class]  # Map index to blood group
    return blood_group, confidence

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Ensure the uploaded file is a BMP image
    if not file.filename.lower().endswith('.bmp'):
        return "Invalid file type. Only BMP files are allowed.", 400

    # Save the file to the TEMP_DIR
    file_path = os.path.join(TEMP_DIR, file.filename)
    file.save(file_path)

    # Make the prediction
    blood_group, confidence = predict_blood_group(file_path)

    # Serve the result in after.html
    file_url = f"/temp/{file.filename}"
    return render_template('after.html', file_url=file_url, result=f"{blood_group} ({confidence:.2f}%)")

# Serve uploaded files
@app.route('/temp/<filename>')
def uploaded_file(filename):
    return send_from_directory(TEMP_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
