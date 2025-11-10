# Importing necessary libraries
import mediapipe as mp
import cv2

# Function to detect hand gestures
def detect_hand_gesture():
    # Initializes Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False)

    # Utility for drawing hand landmarks
    mp_draw = mp.solutions.drawing_utils

    # Opens the webcam for real-time hand tracking
    cap = cv2.VideoCapture(0)

    # Starts webcam capture loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Converts the frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processes the frame to detect hands
        results = hands.process(image)

        # If any hands are detected, draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Displays the webcam feed with detected hand landmarks
        cv2.imshow("Gesture Detection", frame)

        # Press 'q' to quit the program safely
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Releases webcam and closes windows
    cap.release()
    cv2.destroyAllWindows()

# Runs gesture detection when the script is executed directly
if __name__ == "__main__":
    detect_hand_gesture()

