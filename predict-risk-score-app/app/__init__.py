# filepath: /workspaces/py-final/predict-risk-score-app/app/main.py
from flask import Flask, render_template, request
import pandas as pd
from your_model_file import models  # Import your trained models

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = {key: float(value) for key, value in request.form.items()}
    input_df = pd.DataFrame([input_data])

    # Use the model to predict
    model = models['RandomForest_Regressor']  # Replace with your best model
    risk_score = model.predict(input_df)[0]

    # Render the result page
    return render_template('result.html', risk_score=risk_score)

if __name__ == '__main__':
    app.run(debug=True)