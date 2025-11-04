import pandas as pd
import numpy as np

def preprocess_fer_csv(input_path='data/emotion/fer2013.csv'):
    df = pd.read_csv(input_path)
    df.dropna(inplace=True)
    df['emotion'] = df['emotion'].astype(int)
    df['pixels'] = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48).astype('float32') / 255.0)
    return df

if __name__ == "__main__":
    df = preprocess_fer_csv()
    print("Sample record:\n", df.head(1))
