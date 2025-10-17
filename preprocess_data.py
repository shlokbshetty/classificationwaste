# preprocess_data.py
# This script reads images and their corresponding LabelImg XML annotations,
# crops the objects, and saves them into a structured dataset directory.

import os
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_DIR = 'raw_dataset'
PROCESSED_DATA_DIR = 'waste_dataset'
IMAGE_SIZE = (128, 128) # The size to resize cropped images to.

def parse_xml(xml_file):
    """Parses a LabelImg XML file to get bounding box coordinates and labels."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects_data = []
    for member in root.findall('object'):
        label = member.find('name').text
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects_data.append({'label': label, 'box': [xmin, ymin, xmax, ymax]})
    return objects_data

def main():
    """Main function to orchestrate the data preprocessing."""
    print("Starting data preprocessing...")

    # --- 1. Create Processed Data Directories ---
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created directory: {PROCESSED_DATA_DIR}")
    else:
        print(f"Directory already exists: {PROCESSED_DATA_DIR}")

    # --- 2. Find and Process Raw Data ---
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' not found.")
        print("Please create it and place 'wet' and 'dry' subfolders inside.")
        return

    all_image_paths = []
    for class_name in os.listdir(RAW_DATA_DIR):
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(class_dir, filename))

    if not all_image_paths:
        print("Error: No images found in the raw_dataset directory.")
        return

    print(f"Found {len(all_image_paths)} images to process.")
    
    # --- 3. Crop Objects and Save ---
    for image_path in all_image_paths:
        # Construct path to the corresponding XML file
        xml_path = os.path.splitext(image_path)[0] + '.xml'
        
        if not os.path.exists(xml_path):
            print(f"Warning: XML file not found for {os.path.basename(image_path)}. Skipping.")
            continue

        # Read the main image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {os.path.basename(image_path)}. Skipping.")
            continue

        # Parse the XML to get object data
        objects = parse_xml(xml_path)
        
        for i, obj in enumerate(objects):
            label = obj['label']
            box = obj['box']
            
            # Create the destination directory for the class if it doesn't exist
            output_dir = os.path.join(PROCESSED_DATA_DIR, label)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Crop the object from the main image using bounding box coordinates
            cropped_image = image[box[1]:box[3], box[0]:box[2]]
            
            # Resize the cropped image to a standard size
            resized_image = cv2.resize(cropped_image, IMAGE_SIZE)

            # Create a unique filename for the cropped image
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_filename}_{label}_{i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the processed image
            cv2.imwrite(output_path, resized_image)

    print("\nPreprocessing complete!")
    print(f"Cropped and resized images are saved in '{PROCESSED_DATA_DIR}'.")
    print("You are now ready to run the training script.")

if __name__ == '__main__':
    main()