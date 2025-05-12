# Predict Risk Score Application

This project is a web application that allows users to input various financial and personal details to predict a risk score using a machine learning model. The application is built using Flask and provides a user-friendly interface for making predictions.

## Project Structure

```
predict-risk-score-app
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── templates
│   │   └── form.html
│   └── static
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd predict-risk-score-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app/main.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Fill in the form with the required details and submit to get the predicted risk score.

## Features

- User-friendly web interface for inputting data.
- Predicts risk score based on user input using a trained machine learning model.
- Displays the predicted risk score after form submission.
