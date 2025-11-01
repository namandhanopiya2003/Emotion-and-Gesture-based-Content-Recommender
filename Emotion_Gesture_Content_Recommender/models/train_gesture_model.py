import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

data_path = "data/gesture_data.csv"
assert os.path.exists(data_path), f"<!> Gesture data not found at {data_path}"

df = pd.read_csv(data_path, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f">>> Model trained with accuracy: {accuracy*100:.2f}%")

os.makedirs("models", exist_ok=True)
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(">>> Model saved as models/gesture_model.pkl")
