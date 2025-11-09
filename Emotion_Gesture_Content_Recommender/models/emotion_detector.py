# Importing all necessary libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Loads the pre-trained emotion detection model
model = load_model("models/emotion_model.h5")

# List of emotions corresponding to the model’s output classes
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Loads the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Starts capturing video from the webcam
cap = cv2.VideoCapture(0)

print(">>> Emotion detection started (Press Q to quit)")

# Starts reading frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converts the frame to grayscale for better face detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detects faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loops through each detected face
    for x, y, w, h in faces:
        # Extracts the face region from the image
        face = gray[y:y+h, x:x+w]
        # Resizes the face image to 48x48 (model input size) and normalizes it
        resized = cv2.resize(face, (48, 48)) / 255.0
        # Reshapes the image to match the input format of the CNN model
        reshaped = np.reshape(resized, (1, 48, 48, 1))
        # Predicts the emotion using the pre-trained model
        result = model.predict(reshaped, verbose=0)
        # Gets the emotion label with the highest prediction probability
        label = emotion_labels[np.argmax(result)]

        # Draws a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Displays the detected emotion above the face
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Shows the live webcam feed with detected emotions
    cv2.imshow("Emotion Detection", frame)
    # Press 'Q' to stop emotion detection and close the window
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Releases the webcam after exiting the loop
cap.release()
# Closes all OpenCV windows
cv2.destroyAllWindows()
print(">>> Emotion detection stopped.")

