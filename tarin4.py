import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Specify the directory path where the dataset is stored
dataset_directory = "D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\ASL_Alphabet_Dataset\\asl_alphabet_train"

# Preprocess the dataset
image_data = []
labels = []
image_width, image_height = 80, 80

label_encoder = LabelEncoder()  # Create an instance of LabelEncoder

for class_name in os.listdir(dataset_directory):
    class_directory = os.path.join(dataset_directory, class_name)
    if os.path.isdir(class_directory):
        for filename in os.listdir(class_directory):
            if filename.endswith(".jpg"):
                image_path = os.path.join(class_directory, filename)
                label = class_name

                # Load image
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_width, image_height))
                image = tf.keras.preprocessing.image.img_to_array(image)
                image /= 255.0

                # Append image and label to the lists
                image_data.append(image)
                labels.append(label)

# Convert lists to numpy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Encode the labels as integers
labels = label_encoder.fit_transform(labels)

# Define the number of classes
num_classes = len(np.unique(labels))

# Split the dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(image_data, labels, test_size=0.2,
                                                                      random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_images, val_labels)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

model.save("D:\\two-way-sign-language-translator\\two-way-sign-language-translator\\noobchad.h5")