import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import os
import sys

# Constants
IMAGE_SIZE = 224
classes = ['Potato___Late_blight', 'Potato___healthy']

# Set the image path directly here
img_path = "C:\\Users\\HP\\Desktop\\PROJECT\\SMART POTATO FARMING\\test images\\lb3.JPG" 
model_path = "mnet2.h5"  

# Load model
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)

model = keras.models.load_model(model_path)

# Check if image exists
if not os.path.exists(img_path):
    print(f"Error: Image file '{img_path}' not found.")
    sys.exit(1)

def predict_and_display(model, img_path, class_names=classes):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index]

    # Display image and prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence_score:.2%}", fontsize=14, pad=20)
    plt.show()

# Run prediction
predict_and_display(model, img_path)
