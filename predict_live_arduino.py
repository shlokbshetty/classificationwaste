# predict_on_trigger_arduino.py
# This script runs a LIVE camera feed, but only performs
# classification and sends an Arduino signal when you
# press the SPACEBAR.

import cv2
import numpy as np
import serial
import time
import os
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'waste_classifier_model.h5'
CAMERA_ID = 0  # Use 0 for the default built-in laptop webcam
ARDUINO_PORT = 'COM3'  # IMPORTANT: Update to your Arduino's COM port
BAUD_RATE = 9600
CONFIDENCE_THRESHOLD = 0.90
IMAGE_SIZE = (128, 128)
MIN_OBJECT_AREA = 8000  # Min pixel area to be considered an object

# --- Arduino Communication Setup ---
def setup_arduino(port, baud_rate):
    """Initializes and returns the serial connection to the Arduino."""
    try:
        print(f"Connecting to Arduino on {port}...")
        ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        print("Arduino connected successfully.")
        return ser
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        print(f"Please check your COM port: '{port}'")
        print("Running in simulation mode. No signals will be sent.")
        return None

def send_to_arduino(ser, waste_type):
    """Sends a signal ('W' for wet, 'D' for dry) to the Arduino."""
    if ser is not None and ser.is_open:
        if waste_type == 'wet':
            ser.write(b'W')
            print("Sent 'W' (Wet) signal to Arduino")
        elif waste_type == 'dry':
            ser.write(b'D')
            print("Sent 'D' (Dry) signal to Arduino")

# --- Main Application Logic ---
def main():
    """Main function to run the live prediction loop."""
    
    # --- 1. Load Model and Setup Arduino ---
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        class_names = ['dry', 'wet']
        print(f"Classes found: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure the '{MODEL_PATH}' file exists.")
        return
        
    arduino = setup_arduino(ARDUINO_PORT, BAUD_RATE)
    
    # --- 2. Start Camera Feed ---
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera (ID: {CAMERA_ID}).")
        return

    print("\nStarting live prediction... Press SPACE to identify, 'q' to quit.")
    
    # --- 3. State Variables ---
    # These store the *last* prediction result to keep it on screen
    prediction_text = "Press SPACE to identify"
    box_coords = None
    box_color = (255, 255, 255) # White
    
    # --- 4. Prediction Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # We draw on a copy so the original frame is clean for the next prediction
        display_frame = frame.copy()
        
        # --- 5. Check for User Input ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
            
        if key == ord(' '): # SPACEBAR was pressed
            print("Identifying...")
            
            # --- 6. Run Detection & Prediction (ONCE) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            object_detected = False
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > MIN_OBJECT_AREA:
                    object_detected = True
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    box_coords = (x, y, w, h) # Store coords for drawing
                    
                    # --- Predict on the Detected Object ---
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        img = cv2.resize(roi, IMAGE_SIZE)
                        img_array = np.expand_dims(img, axis=0) / 255.0
                        
                        predictions = model.predict(img_array, verbose=0)
                        confidence = np.max(predictions[0])
                        predicted_class_index = np.argmax(predictions[0])
                        predicted_class_name = class_names[predicted_class_index]
                        
                        # --- Update State & Send Signal ---
                        if confidence > CONFIDENCE_THRESHOLD:
                            prediction_text = f"{predicted_class_name.capitalize()} Waste"
                            box_color = (0, 255, 0) # Green
                            # --- Send the signal to Arduino ---
                            print(f"Confident prediction: {prediction_text}")
                            send_to_arduino(arduino, predicted_class_name)
                        else:
                            prediction_text = "Identifying..."
                            box_color = (0, 255, 255) # Yellow
            
            if not object_detected:
                print("No significant object detected.")
                prediction_text = "No Object Found"
                box_coords = None # Clear any previous box

        # --- 7. Draw Last Result (Every Frame) ---
        # This keeps the last result on screen until SPACE is pressed again
        if box_coords:
            x, y, w, h = box_coords
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 3)
        
        # Draw the text
        (text_width, text_height), _ = cv2.getTextSize(prediction_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_x = (display_frame.shape[1] - text_width) // 2
        text_y = 50 # Put text at the top
        cv2.putText(display_frame, prediction_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)

        # --- 8. Show the Final Frame ---
        cv2.imshow('Live Waste Classification', display_frame)

    # --- 9. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
        print("Arduino connection closed.")

if __name__ == '__main__':
    main()
