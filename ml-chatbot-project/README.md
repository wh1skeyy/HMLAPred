# ML Chatbot Project

This project implements a chatbot that predicts risk scores based on user input using a trained machine learning model.

## Project Structure

```
ml-chatbot-project
├── models
│   └── risk_score_model.pkl       # Serialized machine learning model for risk score prediction
├── src
│   ├── app.py                     # Main entry point for the Flask application
│   ├── chatbot.py                 # Logic for handling user input and predicting risk scores
│   └── train_model.py             # Script for training the machine learning model
├── requirements.txt               # List of dependencies for the project
└── README.md                      # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd ml-chatbot-project
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Train the model (if not already trained):**
   Run the `train_model.py` script to train the model and save it as `risk_score_model.pkl`.
   ```
   python src/train_model.py
   ```

4. **Run the application:**
   Start the Flask application by running:
   ```
   python src/app.py
   ```

5. **Access the chatbot:**
   Open your web browser and go to `http://127.0.0.1:5000` to interact with the chatbot.

## Usage

- Input your features into the chatbot interface.
- The chatbot will process the input and return the predicted risk score based on the trained model.

## Dependencies

- Flask
- pandas
- numpy
- scikit-learn

## License

This project is licensed under the MIT License.