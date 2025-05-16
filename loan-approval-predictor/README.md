# LGBM Predictor App

This project is a web application that allows users to input variables for a LightGBM (LGBM) model and receive predictions on the action taken based on those inputs.

## Project Structure

```
loan-approval-predictor
├── src
│   ├── __pycache__
│   │   └── model_utils.cpython-312.pyc  # Compiled Python file, cached bytecode from model_utils.py
│   ├── model
│   │   ├── lgbm_model.pkl               # Serialized LightGBM model file
│   │   └── lgbm.txt                     # LightGBM model text file
│   ├── templates
│   │   └── index.html                   # HTML template for the application interface
│   ├── app.py                           # Main application script
│   └── input_data.csv                   # Input dataset in CSV format
├── README.md                            # Project documentation in Markdown
└── requirements.txt                     # List of project dependencies
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd lgbm-predictor-app
   ```

2. **Install the required dependencies:**
   It is recommended to create a virtual environment before installing the dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   Navigate to the `src` directory and run the `app.py` file via terminal.
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   Open your web browser and go to `http://localhost:5000` or the appropriate URL based on the framework used.

3. **Input Variables:**
   Fill in the required input fields corresponding to the variables used in the LGBM model.

4. **Get Predictions:**
   Submit the form to receive predictions on the action taken based on the input variables.

## LGBM Model Information

The LGBM model used in this application is trained to predict various actions based on input features. Ensure that the input variables match the expected format and data types as required by the model.