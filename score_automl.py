import json
import joblib
import numpy as np
import pandas as pd
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    data = pd.DataFrame(data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    # Make prediction
    result = model.predict(data)
    # Return the predictions as JSON
    return json.dumps({"result": result.tolist()})