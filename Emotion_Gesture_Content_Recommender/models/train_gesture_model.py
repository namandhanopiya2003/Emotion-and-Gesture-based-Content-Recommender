# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Path to the collected gesture data CSV
data_path = "data/gesture_data.csv"
assert os.path.exists(data_path), f"<!> Gesture data not found at {data_path}"

# Loads gesture data into a DataFrame
df = pd.read_csv(data_path, header=None)

# Separates features (hand landmark coordinates) and labels (gesture)
X = df.iloc[:, :-1]                 # All columns except last
y = df.iloc[:, -1]                  # Last column contains gesture labels

# Splits data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializes and trains a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluates the trained model on the test set
accuracy = model.score(X_test, y_test)
print(f">>> Model trained with accuracy: {accuracy*100:.2f}%")

# Ensures the 'models' directory exists for saving the trained model
os.makedirs("models", exist_ok=True)

# Saves the trained model to a pickle file for later use
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(">>> Model saved as models/gesture_model.pkl")

