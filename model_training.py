# model_training.py
# This script trains a CNN to classify images into 'wet_waste' and 'dry_waste'.

import numpy as np
import cv2
import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
IMAGE_SIZE = (128, 128)
DATA_DIR = 'waste_dataset'
MODEL_SAVE_PATH = "waste_classifier_model.h5"

def load_data(data_dir):
    images, labels = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        raise ValueError(f"No class subdirectories found in '{data_dir}'. Please run preprocess_data.py first.")

    label_map = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        image_files = glob(os.path.join(class_path, '*.jpg')) + glob(os.path.join(class_path, '*.png'))
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Load as color image
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label_map[class_name])

    if not images:
        raise ValueError(f"No images found in '{data_dir}'.")

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels, dtype='int32')
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return train_images, train_labels, val_images, val_labels, num_classes

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint]

def train():
    try:
        train_images, train_labels, val_images, val_labels, num_classes = load_data(DATA_DIR)
        
        train_labels = to_categorical(train_labels, num_classes)
        val_labels = to_categorical(val_labels, num_classes)

        print("\n--- Data Shapes ---")
        print(f"Training images shape: {train_images.shape}")
        print(f"Validation images shape: {val_images.shape}")
        print("---------------------\n")

        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3) # 3 channels for color
        model, callbacks_list = build_cnn_model(input_shape, num_classes)
        model.summary()

        print("\n--- Starting Model Training ---")
        model.fit(
            train_images, train_labels, 
            validation_data=(val_images, val_labels), 
            epochs=20, batch_size=32, callbacks=callbacks_list
        )
        print("--- Model Training Finished ---\n")

        model.load_weights(MODEL_SAVE_PATH)
        scores = model.evaluate(val_images, val_labels, verbose=0)
        print(f"Best Validation Accuracy: {scores[1]*100:.2f}%")
        print(f"Model saved to {MODEL_SAVE_PATH}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
    finally:
        K.clear_session()

if __name__ == '__main__':
    train()