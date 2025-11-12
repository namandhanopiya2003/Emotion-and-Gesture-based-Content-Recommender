# Importing all necessary libraries
import cv2
import numpy as np
import pickle
import random
import pandas as pd
import csv
import os
import subprocess

from datetime import datetime
from tensorflow.keras.models import load_model
import mediapipe as mp

# Loads the pre-trained emotion detection model
emotion_model = load_model("models/emotion_model.h5")

# List of emotions corresponding to the model’s output classes
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Loads the pre-trained gesture recognition model
with open("models/gesture_model.pkl", "rb") as f:
    gesture_model = pickle.load(f)

# Possible gestures the model can recognize
gesture_labels = ['like', 'heart', 'peace']

# Loads the content recommendation dataset
df = pd.read_csv("data/recommendation_content.csv")

# Groups content by emotion type for personalized recommendations
emotion_to_content = df.groupby("emotion")["content"].apply(list).to_dict()

# For detecting hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# For detecting faces in webcam feed
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Starts capturing video from webcam
cap = cv2.VideoCapture(0)
print(">>> Project running. Press Q to quit.")

# Initializes default values for emotion and gesture tracking
current_emotion = 'Neutral'
emotion_confidence = 0.5
gesture = 'none'
current_gesture_score = 0.5

# Path to save the session log
log_path = 'data/session_logs.csv'
write_header = not os.path.exists(log_path)

# Starts main webcam loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flips the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)
    # Converts the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        # Crops and resizes the detected face for emotion model input
        face = gray[y:y+h, x:x+w]
        resized = cv2.resize(face, (48, 48)) / 255.0
        reshaped = np.reshape(resized, (1, 48, 48, 1))

        # Predicts emotion using the model
        result = emotion_model.predict(reshaped, verbose=0)[0]
        emotion_idx = int(np.argmax(result))
        current_emotion = emotion_labels[emotion_idx]
        emotion_confidence = float(result[emotion_idx])

        # Draws a rectangle around the detected face and displays emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {current_emotion}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Converts frame to RGB for processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Checks if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draws hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepares input row for gesture model
            row = []
            gesture_magnitude = 0

            # Extracts (x, y, z) coordinates of each landmark
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
                gesture_magnitude += abs(lm.x - 0.5) + abs(lm.y - 0.5)

            # Calculates a basic engagement score based on hand movement
            current_gesture_score = round(min(1.0, gesture_magnitude / 10), 2)

            # Predicts gesture using the trained ML model
            gesture = gesture_model.predict([row])[0]

            # Displays the predicted gesture on the screen
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Selects random content based on detected emotion
    content_list = emotion_to_content.get(current_emotion, ['Explore More!'])
    recommended_content = random.choice(content_list)

    # Identifies content category using emojis in the text
    if any(emoji in recommended_content for emoji in ['🎵', '🎧', '🎤']):
        content_type = 'MUSIC'
    elif any(emoji in recommended_content for emoji in ['📖', '💬', '💡']):
        content_type = 'QUOTE'
    elif any(emoji in recommended_content for emoji in ['🎬', '📹', '🎁']):
        content_type = 'VIDEO'
    else:
        content_type = 'OTHER'

    # Computes the overall "Vibe Score" as a mix of emotion and gesture confidence
    vibe_score = round((emotion_confidence * 0.5) + (current_gesture_score * 0.5), 2)

    # Displays category, recommendation, and vibe score on the frame
    cv2.putText(frame, f'Category: {content_type}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'Content: {recommended_content}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Vibe Score: {vibe_score}', (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

    # Logs session data (emotion, gesture, and recommendation) into CSV file
    with open(log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['timestamp', 'emotion', 'gesture_score', 'gesture', 'engagement', 'recommended_content', 'vibe_score'])
            write_header = False
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            current_emotion,
            round(emotion_confidence, 2),
            gesture,
            current_gesture_score,
            recommended_content,
            vibe_score
        ])

    # Shows the main application window
    cv2.imshow(">>> >> > Emotion + Gesture + Recommendation", frame)

    # Press 'q' to quit the program safely
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Releases the webcam and close all windows after quitting
cap.release()
cv2.destroyAllWindows()
print(">>> Session ended.")

# Automatically run the session analysis script after program ends
print(">>> Running session analysis...\n")
import sys
subprocess.run([sys.executable, "models/analyze_session.py"])
