from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/linear_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Feature list (must match training)
features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

@app.route('/')
def home():
    return render_template("index.html", features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[f]) for f in features]
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]

        result = "✅ Positive (Heart Disease)" if prediction == 1 else "🫀 Negative (No Heart Disease)"
        return render_template("index.html", features=features, prediction=result)

    except Exception as e:
        return render_template("index.html", features=features, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

