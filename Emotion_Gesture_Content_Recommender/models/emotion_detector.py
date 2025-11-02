import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

print(">>> Emotion detection started (Press Q to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        resized = cv2.resize(face, (48, 48)) / 255.0
        reshaped = np.reshape(resized, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label = emotion_labels[np.argmax(result)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(">>> Emotion detection stopped.")
