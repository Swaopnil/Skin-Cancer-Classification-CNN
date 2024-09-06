from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("C:\\Users\\Swapnil Umalkar\\ML_p\\Skin-Cancer-Classification-CNN-master\\Skin-Cancer-Classification-CNN-master\\models\\DenseNet201.keras")

# Function to preprocess image
def preprocess_image(image):
    # Resize and preprocess image
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image from request
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), -1)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))[0]
    
    if prediction > 0.5:
        predicted_class = "malignant"
    else:
        predicted_class = "benign"

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
