import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)
print("Accuracy:", accuracy)

# Save Model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved successfully!")