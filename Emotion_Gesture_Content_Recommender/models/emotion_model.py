import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

def train_emotion_model(data_dir="data/emotion/fer2013"):
    img_size = 48
    batch_size = 64

    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1) 

    train_gen = datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset='validation'
    )

    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

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

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=15)

    model.save("models/emotion_model.h5")
    print("[INFO] Model trained and saved to models/emotion_model.h5")

    return model

if __name__ == "__main__":
    train_emotion_model()
