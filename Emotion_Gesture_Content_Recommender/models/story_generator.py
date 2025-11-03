import random

vibe_stories = {
    "happy": ["Feeling the sunshine!", "Life is beautiful today!", "Smiles all around!"],
    "sad": ["Taking it one step at a time.", "Rainy days and thoughts.", "Need a warm hug today."],
    "angry": ["Trying to cool down.", "Patience is a virtue!", "Letting the steam out."],
    "neutral": ["Just another day!", "Balanced vibes.", "Flowing through the moment."]
}

def generate_story(emotion_label):
    return random.choice(vibe_stories.get(emotion_label.lower(), ["Enjoying the moment."]))
