import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a list to store the images of each gesture
gesture_images = []

# Iterate over the folders in the D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\archive\\Train_Alphabet
for gesture_folder in os.listdir('D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\archive\\Train_Alphabet'):
    # Iterate over the images in the gesture folder
    for image in os.listdir('D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\archive\\Train_Alphabet/{}'.format(gesture_folder)):
        # Read the image
        image = cv2.imread('D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\archive\\Train_Alphabet/{}/{}'.format(gesture_folder, image))

        # Convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to 224x224
        image = cv2.resize(image, (224, 224))

        # Add the image to the list of images for the gesture
        gesture_images.append(image)

# Create a CNN model to train the gesture recognizer
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(gesture_images, np.array([0, 1, 2, 3, 4]), epochs=10)

# Save the model
model.save('gesture_recognizer.h5')

# Close the webcam
cap.release()
