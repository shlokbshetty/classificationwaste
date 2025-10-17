# predict_image.py
# This script loads the trained waste classification model and predicts on a single image file.

import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
IMAGE_SIZE = (128, 128)
MODEL_PATH = "waste_classifier_model.h5"
DATA_DIR = 'waste_dataset'

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR) # Load color image
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")
    
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

def main():
    parser = argparse.ArgumentParser(description="Predict waste type from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Run training first.")
        return
        
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    try:
        class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        if not class_names: raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: Could not find class subdirectories in '{DATA_DIR}'.")
        return

    try:
        processed_image = preprocess_image(args.image_path)
    except FileNotFoundError as e:
        print(e)
        return

    print("Making prediction...")
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100

    print("\n--- Prediction Result ---")
    print(f"Predicted Waste Type: {predicted_class_name.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")
    print("-------------------------\n")

if __name__ == '__main__':
    main()
