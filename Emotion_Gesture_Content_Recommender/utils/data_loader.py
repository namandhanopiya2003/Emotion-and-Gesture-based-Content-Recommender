import pandas as pd

def load_mock_user_data(path='data/mock_user_data.csv'):
    
    # Loads CSV into a DataFrame
    df = pd.read_csv(path)

    # Extracts features and labels
    features = df[['likes', 'comments', 'gesture_score', 'emotion_id']]
    labels = df['content_type']
    
    return features, labels

