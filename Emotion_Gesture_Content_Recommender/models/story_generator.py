# Importing necessary library for random selection
import random

# Dictionary mapping emotions to example "vibe stories"
vibe_stories = {
    "happy": ["Feeling the sunshine!", "Life is beautiful today!", "Smiles all around!"],
    "sad": ["Taking it one step at a time.", "Rainy days and thoughts.", "Need a warm hug today."],
    "angry": ["Trying to cool down.", "Patience is a virtue!", "Letting the steam out."],
    "neutral": ["Just another day!", "Balanced vibes.", "Flowing through the moment."]
}

# Function to generate a story based on detected emotion
def generate_story(emotion_label):
    # Selects a random story corresponding to the emotion
    return random.choice(vibe_stories.get(emotion_label.lower(), ["Enjoying the moment."]))

