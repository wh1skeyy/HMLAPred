import pickle
import numpy as np

class RiskScoreChatbot:
    def __init__(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def predict_risk_score(self, input_features):
        # Ensure input features are in the correct format
        input_array = np.array(input_features).reshape(1, -1)
        risk_score = self.model.predict(input_array)
        return risk_score[0]

def main():
    # Example usage
    model_path = 'models/risk_score_model.pkl'
    chatbot = RiskScoreChatbot(model_path)

    # Simulated user input
    user_input = [/* input features here */]
    risk_score = chatbot.predict_risk_score(user_input)
    print(f'Predicted Risk Score: {risk_score}')

if __name__ == '__main__':
    main()