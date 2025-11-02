def calculate_vibe_score(emotion_id, gesture_score, session_time, interactions):
    score = (emotion_id + gesture_score) * 10 + session_time * 0.5 + interactions * 2
    return round(score, 2)
