# Importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Function to train a simple content recommender model
def train_recommender(data_path='data/mock_user_data.csv'):
    # Loads user interaction dataset
    df = pd.read_csv(data_path)

    # Features for prediction: likes, comments, gesture score, and emotion ID
    X = df[['likes', 'comments', 'gesture_score', 'emotion_id']]

    # Target variable: type of content to recommend
    y = df['content_type']

    # Initializes the Random Forest classifier
    clf = RandomForestClassifier()

    # Trains the model on the user interaction data
    clf.fit(X, y)

    # Returns the trained model
    return clf

# Function to predict content recommendation for a new user input
def recommend_content(model, user_input):
    # Predicts the most suitable content type based on input features
    return model.predict([user_input])[0]

