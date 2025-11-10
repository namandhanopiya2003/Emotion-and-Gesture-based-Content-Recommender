# Importing necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Loads the pre-trained gesture recognition model
with open("models/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initializes hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Opens the webcam for real-time gesture prediction
cap = cv2.VideoCapture(0)

# Checks if webcam opened successfully
if not cap.isOpened():
    print("<!> Could not open webcam.")
    exit()

print(">>> Predicting gestures in real-time. Press 'Q' to quit.")

# Starts the main loop for real-time gesture prediction
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flips frame horizontally for mirror-like view
    frame = cv2.flip(frame, 1)
    # Converts frame from BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processes the frame to detect hand landmarks
    results = hands.process(rgb)

    # If any hands are detected, process landmarks for prediction
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draws detected hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collects all landmark coordinates (x, y, z) into a single list
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Converts landmark list to a NumPy array for model input
            input_data = np.array(row).reshape(1, -1)
            # Predicts gesture using the pre-trained model
            predicted_label = model.predict(input_data)[0]

            # Displays predicted gesture on the frame
            cv2.putText(frame, f"Gesture: {predicted_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Shows the webcam feed with gesture predictions
    cv2.imshow("Gesture Prediction", frame)

    # Press 'q' to quit the program safely
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Releases the webcam and closes windows
cap.release()
cv2.destroyAllWindows()

