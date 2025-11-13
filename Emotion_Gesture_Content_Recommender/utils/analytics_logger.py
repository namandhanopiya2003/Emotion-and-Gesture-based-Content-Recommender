import csv
import os
from datetime import datetime

# Function to log user session data into a CSV file
def log_session_data(emotion, gesture, vibe_score, session_time, content_type, path="data/session_log.csv"):
    
    # Checks if the file already exists
    file_exists = os.path.isfile(path)

    # Opens the CSV file in append mode
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Writes header row if the file is being created for the first time
        if not file_exists:
            writer.writerow(["timestamp", "emotion", "gesture", "vibe_score", "session_time", "content_type"])
        # Appends the session data as a new row    
        writer.writerow([datetime.now(), emotion, gesture, vibe_score, session_time, content_type])
    # Prints confirmation message    
    print(f"Logged session to {path}")
