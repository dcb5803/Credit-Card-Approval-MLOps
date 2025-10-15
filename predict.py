import joblib
import pandas as pd

def predict(input_dict):
    model = joblib.load("model.joblib")
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]
