import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_recommender(data_path='data/mock_user_data.csv'):
    df = pd.read_csv(data_path)
    X = df[['likes', 'comments', 'gesture_score', 'emotion_id']]
    y = df['content_type']
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

def recommend_content(model, user_input):
    return model.predict([user_input])[0]
