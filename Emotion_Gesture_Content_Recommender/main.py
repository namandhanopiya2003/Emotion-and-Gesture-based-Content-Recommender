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

emotion_model = load_model("models/emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

with open("models/gesture_model.pkl", "rb") as f:
    gesture_model = pickle.load(f)
gesture_labels = ['like', 'heart', 'peace']

df = pd.read_csv("data/recommendation_content.csv")
emotion_to_content = df.groupby("emotion")["content"].apply(list).to_dict()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print(">>> Project running. Press Q to quit.")

current_emotion = 'Neutral'
emotion_confidence = 0.5
gesture = 'none'
current_gesture_score = 0.5

log_path = 'data/session_logs.csv'
write_header = not os.path.exists(log_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        resized = cv2.resize(face, (48, 48)) / 255.0
        reshaped = np.reshape(resized, (1, 48, 48, 1))
        result = emotion_model.predict(reshaped, verbose=0)[0]
        emotion_idx = int(np.argmax(result))
        current_emotion = emotion_labels[emotion_idx]
        emotion_confidence = float(result[emotion_idx])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {current_emotion}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = []
            gesture_magnitude = 0
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
                gesture_magnitude += abs(lm.x - 0.5) + abs(lm.y - 0.5)
            current_gesture_score = round(min(1.0, gesture_magnitude / 10), 2)
            gesture = gesture_model.predict([row])[0]
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    content_list = emotion_to_content.get(current_emotion, ['Explore More!'])
    recommended_content = random.choice(content_list)

    if any(emoji in recommended_content for emoji in ['🎵', '🎧', '🎤']):
        content_type = 'MUSIC'
    elif any(emoji in recommended_content for emoji in ['📖', '💬', '💡']):
        content_type = 'QUOTE'
    elif any(emoji in recommended_content for emoji in ['🎬', '📹', '🎁']):
        content_type = 'VIDEO'
    else:
        content_type = 'OTHER'

    vibe_score = round((emotion_confidence * 0.5) + (current_gesture_score * 0.5), 2)

    cv2.putText(frame, f'Category: {content_type}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'Content: {recommended_content}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Vibe Score: {vibe_score}', (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

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

    cv2.imshow(">>> >> > Emotion + Gesture + Recommendation", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(">>> Session ended.")

print(">>> Running session analysis...\n")
import sys
subprocess.run([sys.executable, "models/analyze_session.py"])
