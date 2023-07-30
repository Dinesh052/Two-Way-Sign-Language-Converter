import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from pathlib import Path

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Get the list of gesture folders
gesture_folders = []
for entry in Path("D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\archive\\Train_Alphabet").iterdir():
  if entry.is_dir():
    gesture_folders.append(entry)

# Initialize the CUDA context
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
  tf.config.experimental.set_memory_growth(gpus[0], True)

# Create a TensorFlow model
model = Sequential([
  Flatten(input_shape=(2048,)),
  Dense(128, activation="relu"),
  Dropout(0.2),
  Dense(64, activation="relu"),
  Dropout(0.2),
  Dense(len(gesture_folders), activation="softmax")
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model
images = []
for gesture_folder in gesture_folders:
  for entry in gesture_folder.iterdir():
    if entry.is_file() and entry.suffix == ".jpg":
      images.append(cv2.imread(str(entry)))
  model.fit(images, np.zeros(len(gesture_folders)), epochs=10, batch_size=32, verbose=0)

# Save the model
model.save("model.h5")
