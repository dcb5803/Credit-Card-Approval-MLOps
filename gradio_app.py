import gradio as gr
import pandas as pd
import joblib

model = joblib.load("model.joblib")

def predict_credit(age, income, education, marital_status):
    df = pd.DataFrame([{
        "age": age,
        "income": income,
        "education": education,
        "marital_status": marital_status
    }])
    pred = model.predict(df)[0]
    return "Approved ✅" if pred == 1 else "Rejected ❌"

demo = gr.Interface(
    fn=predict_credit,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Income"),
        gr.Dropdown(["HighSchool", "Bachelors", "Masters"], label="Education"),
        gr.Dropdown(["Single", "Married", "Divorced"], label="Marital Status")
    ],
    outputs="text",
    title="Credit Card Approval Predictor"
)

demo.launch()
