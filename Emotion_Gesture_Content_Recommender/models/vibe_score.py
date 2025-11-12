# Function to compute the overall "Vibe Score" for a user session
def calculate_vibe_score(emotion_id, gesture_score, session_time, interactions):
    score = (emotion_id + gesture_score) * 10 + session_time * 0.5 + interactions * 2
    # Returns the final score rounded to two decimal places
    return round(score, 2)

