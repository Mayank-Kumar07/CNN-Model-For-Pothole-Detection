import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained CNN model
model_path = 'D:/vs jupyter/neuralnetworks/pothole_detection_model.h5'
model = load_model(model_path)



# Define image dimensions
image_height, image_width = 224, 224

def load_and_predict(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(image_height, image_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Class label
    if prediction > 0.5:
        return "No Pothole Detected"
    else:
        return "Pothole Detected"

# Test the model on new images
#image_dir = r"neuralnetworks\test\potholes"
image_dir = r"neuralnetworks\test\not_potholes"

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        result = load_and_predict(image_path)
        print(f"Image: {filename}, Prediction: {result}")
