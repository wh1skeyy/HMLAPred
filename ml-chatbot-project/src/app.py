from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('models/risk_score_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert input data to DataFrame
    input_data = pd.DataFrame(data, index=[0])
    
    # Make prediction
    risk_score = model.predict(input_data)
    
    return jsonify({'RiskScore': risk_score[0]})

if __name__ == '__main__':
    app.run(debug=True)