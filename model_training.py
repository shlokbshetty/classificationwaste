# model_training.py (Updated with Data Augmentation)
# This script trains a CNN to classify waste images.
# It now uses data augmentation to improve model robustness.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
DATA_DIR = 'waste_dataset'
MODEL_PATH = 'waste_classifier_model.h5'
PLOT_PATH = 'training_history.png'
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100 # Increased epochs to run for more time

def create_cnn_model(input_shape, num_classes):
    """Creates and compiles the CNN model."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25), # Added dropout to regularize earlier

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25), # Added dropout to regularize earlier

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    
    # Using a slower learning rate to allow for more stable learning
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    """Plots the training and validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(PLOT_PATH)
    print(f"Training history plot saved to {PLOT_PATH}")
    # plt.show() # Uncomment this line if you want to see the plot immediately

def main():
    """Main function to orchestrate the model training."""
    print("Starting model training...")
    
    # --- 1. Check for data directory ---
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: The '{DATA_DIR}' directory is empty or does not exist.")
        print("Please run the 'preprocess_data.py' script first to create the dataset.")
        return

    # --- 2. Create Data Generators for Augmentation ---
    print("Setting up data augmentation...")
    # For training data: apply random transformations
    # We also specify a validation split to separate training and testing data
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Use 20% of the data for validation
    )

    # --- 3. Prepare Iterators ---
    print("Preparing data iterators...")
    # Create a generator for training data
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training' # Specify this is for training
    )

    # Create a generator for validation data
    validation_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation' # Specify this is for validation
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Found {train_generator.samples} images belonging to {num_classes} classes.")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    print(f"Using {train_generator.samples} images for training and {validation_generator.samples} for validation.")


    # --- 4. Create and Train Model ---
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    model = create_cnn_model(input_shape, num_classes)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # Increased patience to give the model more time to improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nStarting training with augmented data...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    print("\nTraining complete!")
    if history.history.get('val_accuracy'):
        best_val_acc = max(history.history['val_accuracy']) * 100
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_PATH}")

    # --- 5. Plotting and Clean up ---
    plot_history(history)
    K.clear_session()

if __name__ == '__main__':
    main()

