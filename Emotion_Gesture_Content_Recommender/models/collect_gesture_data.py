import cv2
import mediapipe as mp
import csv
import os

GESTURE_LABEL = "peace"

os.makedirs("data", exist_ok=True)
save_path = "data/gesture_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("<!> ERROR: Could not open webcam.")
    exit()

print(f">>> Collecting gesture data for label: '{GESTURE_LABEL}'")
print(">>> Press 'Q' to quit.\n")

with open(save_path, mode='a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("<!> Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append(GESTURE_LABEL)
                writer.writerow(row)

        cv2.imshow("Collecting Gesture Data", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(">>> Data collection finished.")
