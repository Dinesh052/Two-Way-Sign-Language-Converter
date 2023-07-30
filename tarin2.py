import glob

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

# Initialize MediaPipe hands model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a list to store the training data
training_data = []

# Load the class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Iterate over the gestures
for gesture in classNames:
    # Get the images for the gesture
    images = glob.glob('D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\archive\\Train_Alphabet\\/' + gesture + '/*.png')

    # Loop over the images
    for image in images:
        # Read the image
        img = cv2.imread(image)

        # Get the hand landmark prediction
        result = hands.process(img)

        # Post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * img.shape[1])
                    lmy = int(lm.y * img.shape[0])

                    landmarks.append([lmx, lmy])

            # Add the training data to the list
            training_data.append([landmarks, gesture])

# Create a model
model = Sequential()
model.add(Flatten(input_shape=(21,)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(classNames), activation='softmax'))

# Compile the model
with tf.device('/gpu:1'):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training_data, epochs=10)

# Save the model
model.save('mp_hand_gesture')
