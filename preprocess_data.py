# preprocess_data.py
# This script converts an object detection dataset (images + XML files from labelImg)
# into a classification dataset with a clean folder structure.

import os
import cv2
import xml.et.ree.ElementTree as ET
from glob import glob

# --- Configuration ---
SOURCE_DATA_DIR = 'train' 
DESTINATION_DATA_DIR = 'waste_dataset'
CLASSES = ['wet_waste', 'dry_waste']

def process_dataset():
    """
    Reads XML files, crops corresponding images based on bounding box data,
    and saves them into the correct class subfolders.
    """
    print("Starting dataset preprocessing...")
    print(f"Source Directory: '{SOURCE_DATA_DIR}'")
    print(f"Destination Directory: '{DESTINATION_DATA_DIR}'")

    # 1. Create destination directories
    if not os.path.exists(DESTINATION_DATA_DIR):
        os.makedirs(DESTINATION_DATA_DIR)
    for class_name in CLASSES:
        class_path = os.path.join(DESTINATION_DATA_DIR, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

    # 2. Find all XML annotation files
    xml_files = glob(os.path.join(SOURCE_DATA_DIR, '*.xml'))
    if not xml_files:
        print(f"Error: No .xml files found in '{SOURCE_DATA_DIR}'. Please check the path.")
        return

    print(f"\nFound {len(xml_files)} XML files to process.")
    
    processed_count = 0
    # 3. Loop through each XML file
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            image_filename = root.find('filename').text
            image_path = os.path.join(SOURCE_DATA_DIR, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found for {xml_file}. Skipping.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue

            # Find all 'object' tags
            for member in root.findall('object'):
                class_name = member.find('name').text
                if class_name not in CLASSES:
                    print(f"Warning: Unknown class '{class_name}' in {xml_file}. Skipping.")
                    continue

                bndbox = member.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Crop the image and save it
                cropped_image = image[ymin:ymax, xmin:xmax]
                base_filename = os.path.splitext(os.path.basename(image_filename))[0]
                output_filename = f"{base_filename}_cropped_{processed_count}.jpg"
                output_path = os.path.join(DESTINATION_DATA_DIR, class_name, output_filename)
                cv2.imwrite(output_path, cropped_image)
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing file {xml_file}: {e}")

    print(f"\nPreprocessing complete. Saved {processed_count} cropped images into '{DESTINATION_DATA_DIR}'.")
    print("You can now run the training script.")

if __name__ == '__main__':
    process_dataset()

