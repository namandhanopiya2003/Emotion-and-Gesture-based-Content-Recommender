# Importing all necessary libraries
import cv2
import mediapipe as mp
import csv
import os

# Sets the gesture label for which data will be collected
GESTURE_LABEL = "peace"

# Creates a folder named "data" if it doesn’t already exist
os.makedirs("data", exist_ok=True)
# File path where collected gesture data will be saved
save_path = "data/gesture_data.csv"

# Initializes MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Opens the webcam
cap = cv2.VideoCapture(0)

# Checks if webcam opened successfully
if not cap.isOpened():
    print("<!> ERROR: Could not open webcam.")
    exit()

# Displays basic information for the user
print(f">>> Collecting gesture data for label: '{GESTURE_LABEL}'")
print(">>> Press 'Q' to quit.\n")

# Opens the CSV file to store hand landmark data
with open(save_path, mode='a', newline='') as f:
    writer = csv.writer(f)

    # Starts capturing frames continuously
    while True:
        ret, frame = cap.read()
        if not ret:
            print("<!> Failed to read frame.")
            break

        # Flips the frame horizontally for a mirror-like display
        frame = cv2.flip(frame, 1)
        # Converts the frame from BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processes the frame to detect hand landmarks
        results = hands.process(rgb)

        # Checks if any hand landmarks were detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draws detected hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Creates an empty list to store landmark coordinates
                row = []
                for lm in hand_landmarks.landmark:
                    # Adds x, y, z coordinates of each landmark point
                    row.extend([lm.x, lm.y, lm.z])
                # Appends the gesture label at the end of the row
                row.append(GESTURE_LABEL)
                # Writes the row (gesture data) to the CSV file
                writer.writerow(row)

        # Displays the live camera feed while collecting data
        cv2.imshow("Collecting Gesture Data", frame)
        # Quits the program if 'Q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Releases the webcam after exiting the loop
cap.release()
# Closes all OpenCV windows
cv2.destroyAllWindows()
print(">>> Data collection finished.")

