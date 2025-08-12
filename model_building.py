import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv("heart.csv")

# Select only numeric features
X = df[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]]
y = df["HeartDisease"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train simple logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/linear_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model training complete and saved.")
