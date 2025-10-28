# Voting Ensemble for Heart Disease Prediction

A simple machine learning project that predicts heart disease using a **Voting Ensemble** of multiple classifiers on the **UCI Heart Disease dataset**.

---

## How It Works

The project trains a voting ensemble combining:
- Logistic Regression  
- K-Nearest Neighbors  
- Support Vector Machine (SVM)  
- Random Forest  
- Gradient Boosting  
- AdaBoost  

It then saves the trained model and scaler for later predictions.

---

## Files

| File | Description |
|------|--------------|
| `ensemble_model.py` | Trains the voting ensemble and saves the model (`voting_model.pkl`) and scaler (`scaler.pkl`). |
| `test.py` | Loads the saved model and makes predictions on new input data. |
| `requirements.txt` | Lists the required Python libraries. |

---
