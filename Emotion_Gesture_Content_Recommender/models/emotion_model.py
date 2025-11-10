# Importing all necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Defines a function to train the emotion recognition model
def train_emotion_model(data_dir="data/emotion/fer2013"):
    # Sets image size and batch size for training
    img_size = 48
    batch_size = 64

    # Defines paths for training and testing data
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    # Prepares image data generator for normalization and validation split
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1) 

    # Loads and preprocesses training images
    train_gen = datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )

    # Loads and preprocesses validation images
    val_gen = datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='validation'
    )

    # Loads and preprocesses test images
    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    # Builds a simple CNN model for emotion classification
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(train_gen.num_classes, activation='softmax')
    ])

    # Compiles the model with optimizer, loss function, and evaluation metric
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains the model using training and validation datasets
    model.fit(train_gen, validation_data=val_gen, epochs=15)

    # Saves the trained model for future use
    model.save("models/emotion_model.h5")
    print("[INFO] Model trained and saved to models/emotion_model.h5")

    return model

# Runs the training function when the script is executed directly
if __name__ == "__main__":
    train_emotion_model()

