import pandas as pd

def load_mock_user_data(path='data/mock_user_data.csv'):
    df = pd.read_csv(path)
    features = df[['likes', 'comments', 'gesture_score', 'emotion_id']]
    labels = df['content_type']
    return features, labels
