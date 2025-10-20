
import cv2
import numpy as np
import serial
import time
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'waste_classifier_model.h5'
# IMPORTANT: Update this to your phone's camera URL from DroidCam/Iriun
CAMERA_URL = "http://YOUR_PHONE_IP_ADDRESS:PORT/video" 
# IMPORTANT: Update this to the correct COM port for your Arduino
ARDUINO_PORT = 'COM3' 
BAUD_RATE = 9600
CONFIDENCE_THRESHOLD = 0.90 # Only send a signal if confidence is > 90%
IMAGE_SIZE = (128, 128)
COOLDOWN_PERIOD = 5 # (seconds) - Prevents sending signals too frequently
# Increased minimum area to be considered an object to avoid classifying the background
MIN_OBJECT_AREA = 8000 

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
        # Assuming the classes are in alphabetical order ('dry', 'wet')
        class_names = ['dry', 'wet'] 
        print(f"Classes found: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure the '{MODEL_PATH}' file exists.")
        return

    arduino = setup_arduino(ARDUINO_PORT, BAUD_RATE)
    
    # --- 2. Start Camera Feed ---
    # To use a built-in webcam, change CAMERA_URL to 0
    # cap = cv2.VideoCapture(0) 
    cap = cv2.VideoCapture(CAMERA_URL)
    
    if not cap.isOpened():
        print("Error: Could not open camera feed.")
        print(f"Please check if the URL '{CAMERA_URL}' is correct and the camera is active.")
        return

    print("\nStarting live prediction... Press 'q' to quit.")
    
    # --- 3. Cooldown and State Management ---
    last_signal_time = 0
    last_prediction = None
    
    # --- 4. Prediction Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Ending stream.")
            break

        # --- 5. Dynamic Object Detection ---
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Use a threshold to create a binary mask
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        object_detected = False
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only proceed if the contour is reasonably large
            if cv2.contourArea(largest_contour) > MIN_OBJECT_AREA:
                object_detected = True
                # Get the bounding box for the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # --- 6. Predict on the Detected Object ---
                # Crop the object from the frame
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    # Preprocess the cropped image
                    img = cv2.resize(roi, IMAGE_SIZE)
                    img_array = np.expand_dims(img, axis=0) / 255.0
                    
                    # Make a prediction
                    predictions = model.predict(img_array, verbose=0)
                    confidence = np.max(predictions[0])
                    predicted_class_index = np.argmax(predictions[0])
                    predicted_class_name = class_names[predicted_class_index]
                    
                    # --- 7. Update Display and Send Signal ---
                    if confidence > CONFIDENCE_THRESHOLD:
                        prediction_text = f"{predicted_class_name.capitalize()} Waste"
                        # Draw a green box for confident predictions
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        
                        # Check cooldown to prevent spamming signals
                        current_time = time.time()
                        if predicted_class_name != last_prediction or (current_time - last_signal_time) > COOLDOWN_PERIOD:
                            send_to_arduino(arduino, predicted_class_name)
                            last_prediction = predicted_class_name
                            last_signal_time = current_time
                    else:
                        prediction_text = "Identifying..."
                        # Draw a yellow box for uncertain predictions
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        last_prediction = None # Reset last prediction if confidence is low
                
                # Display the prediction text above the box
                cv2.putText(frame, prediction_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if not object_detected:
            # If no significant object is found, display a prompt
            prompt_text = "Present Waste to Camera"
            (text_width, text_height), _ = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            text_x = (frame.shape[1] - text_width) // 2
            text_y = (frame.shape[0] + text_height) // 2
            cv2.putText(frame, prompt_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            last_prediction = None # Reset state if no object is seen

        # --- 8. Show the Final Frame ---
        cv2.imshow('Live Waste Classification', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 9. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
        print("Arduino connection closed.")

if __name__ == '__main__':
    main()

