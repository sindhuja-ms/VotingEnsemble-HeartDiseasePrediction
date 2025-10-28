import pandas as pd
import joblib

model = joblib.load("voting_model.pkl")
scaler = joblib.load("scaler.pkl")

test_data = pd.DataFrame([{
    'age': 54,
    'sex': 1,
    'cp': 2,
    'trestbps': 130,
    'chol': 246,
    'fbs': 0,
    'restecg': 0,
    'thalach': 150,
    'exang': 0,
    'oldpeak': 1.3,
    'slope': 2,
    'ca': 0,
    'thal': 2
}])


scaled_data = scaler.transform(test_data)
prediction = model.predict(scaled_data)

print("Prediction (0 = No heart disease, 1 = Heart disease):", prediction[0])
