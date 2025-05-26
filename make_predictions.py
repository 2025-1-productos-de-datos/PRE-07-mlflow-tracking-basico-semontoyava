"""Prediccion script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset.

$ python3 make_predictions.py


"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## Debe verificarse el run_id del modelo que se quiere cargars
## Se puede obtener el run_id desde la interfaz de MLflow
logged_model = "runs:/9d7ef75ea5ba4ad38f0beb59d2a7a7dd/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)

print(y)
