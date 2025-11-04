import csv
import os
from datetime import datetime

def log_session_data(emotion, gesture, vibe_score, session_time, content_type, path="data/session_log.csv"):
    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "emotion", "gesture", "vibe_score", "session_time", "content_type"])
        writer.writerow([datetime.now(), emotion, gesture, vibe_score, session_time, content_type])
    print(f"Logged session to {path}")
