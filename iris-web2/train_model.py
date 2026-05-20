# Upload Dataset ZIP


from google.colab import files
uploaded = files.upload()


# Extract ZIP File


import zipfile
import os
import shutil


zip_file_name = list(uploaded.keys())[0]


with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall()


print("Dataset extracted successfully!")


# Set Dataset Path


dataset_path = "dataset"


print("Dataset folders:", os.listdir(dataset_path))


# Remove unwanted system folders if exists
if ".ipynb_checkpoints" in os.listdir(dataset_path):
    shutil.rmtree(os.path.join(dataset_path, ".ipynb_checkpoints"))


# Import Libraries


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


print("TensorFlow Version:", tf.__version__)


# Image Settings


IMG_SIZE = 128
BATCH_SIZE = 16


# Data Augmentation + Loading


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)


validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


print("Classes:", train_generator.class_indices)


# Build CNN Model


model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),


    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),


    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),


    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),


    layers.Flatten(),


    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),


    layers.Dense(2, activation='softmax')
])


print("Model built successfully!")


# Compile Model


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


print("Model compiled successfully!")


# Early Stopping


early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)


# Train Model


print("Training started...")


history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)


print("Training finished!")


# Save Only the Weights (Guarantees Cross-Version Compatibility)
model.save_weights("device_model.weights.h5")
print("Model weights saved successfully!")


# Download Weights File
files.download("device_model.weights.h5")
print("Download started! Use 'device_model.weights.h5' in your local app.py.")

