from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

from utils import prepare_data


app = Flask(__name__)


# Загрузка модели и кодировщиков
with open('app/models/model_and_encoders.pkl', 'rb') as f:
    model, le_zipcode, le_state, onehot_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prepared_data = prepare_data(df, le_zipcode, le_state, onehot_encoder)
    y_pred_log = model.predict(prepared_data)
    y_pred = np.exp(y_pred_log)
    return jsonify({'prediction': y_pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
