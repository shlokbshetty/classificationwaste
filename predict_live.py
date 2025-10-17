# predict_live.py
# This script uses OpenCV to capture live video from a webcam,
# runs the waste classification model on each frame, and displays the results.

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
MODEL_PATH = "waste_classifier_model.h5"
DATA_DIR = 'waste_dataset'
IMAGE_SIZE = (128, 128)

def main():
    # --- 1. Load Model and Class Names ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please run training first.")
        return
    
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    try:
        class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        if not class_names: raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: Could not find class subdirectories in '{DATA_DIR}'.")
        return
    print(f"Classes found: {class_names}")

    # --- 2. Initialize Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting live prediction... Press 'q' to quit.")
    
    while True:
        # --- 3. Capture and Preprocess Frame ---
        ret, frame = cap.read()
        if not ret:
            break

        # Define a region of interest (ROI) for the user to place the object
        h, w, _ = frame.shape
        roi_size = 300
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # Extract the ROI
        roi = frame[y1:y2, x1:x2]

        # Preprocess the ROI for the model
        img_resized = cv2.resize(roi, IMAGE_SIZE)
        img_normalized = img_resized.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # --- 4. Make Prediction ---
        prediction = model.predict(img_expanded)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction) * 100

        # --- 5. Display Results on Frame ---
        label = f"{predicted_class_name.replace('_', ' ').title()}: {confidence:.2f}%"
        color = (0, 255, 0) # Green for text
        
        # Draw the ROI box on the main frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Put the label text above the ROI box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Live Waste Classification', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 6. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Prediction stopped.")

if __name__ == '__main__':
    main()