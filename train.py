import pandas as pd
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import mlflow
import joblib

# Embedded dataset
data = """
age,income,education,marital_status,approved
25,50000,Bachelors,Single,1
45,120000,Masters,Married,1
22,30000,HighSchool,Single,0
35,80000,Bachelors,Divorced,1
29,40000,HighSchool,Single,0
"""

df = pd.read_csv(StringIO(data))
X = df.drop("approved", axis=1)
y = df["approved"]

categorical = ["education", "marital_status"]
numeric = ["age", "income"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical),
    ("num", "passthrough", numeric)
])

pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("credit_card_approval")
with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(pipeline, "model")
    joblib.dump(pipeline, "model.joblib")
