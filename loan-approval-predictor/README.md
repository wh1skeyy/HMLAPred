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
   Right-click on /loan-approval-predictor, choose Terminal
   Run:
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application:**
   Navigate to the `src` directory and run the `app.py` file.
   Right-click on /src, choose Terminal
   Run:
   ```
   python app.py
   ```

2. **Access the web interface:**
   Open your web browser and go to `http://localhost:5000` or the appropriate URL based on the framework used.
   On GitHub Codespaces, Navigate to Ports Tab on the Right-hand side of the Terminal, choose the Forwarded Address with Port 5000.

3. **Input Variables:**
   Fill in the required input fields corresponding to the variables used in the LGBM model.

4. **Get Predictions:**
   Submit the form to receive predictions on the action taken based on the input variables.

## LGBM Model Information

The LGBM model used in this application is trained to predict various actions based on input features. Ensure that the input variables match the expected format and data types as required by the model.

## Description of the variables on the web application form and their possible values
* Any variables that have possible values as float: any float number

* Variables that only have certain values:

  +  purchaser_type -- Type of entity purchasing a covered loan from the institution
  ```
  Values:
  0 - Not applicable
  1 - Fannie Mae
  2 - Ginnie Mae
  3 - Freddie Mac
  4 - Farmer Mac
  5 - Private securitizer
  6 - Commercial bank, savings bank, or savings association
  71 - Credit union, mortgage company, or finance company
  72 - Life insurance company
  8 - Affiliate institution
  9 - Other type of purchaser
  ```

  +  loan_type -- The type of covered loan or application
  ```
  Values:
  1 - Conventional (not insured or guaranteed by FHA, VA, RHS, or FSA)
  2 - Federal Housing Administration insured (FHA)
  3 - Veterans Affairs guaranteed (VA)
  4 - USDA Rural Housing Service or Farm Service Agency guaranteed (RHS or FSA)
  ```
  + loan_purpose -- The purpose of covered loan or application
  ```
  Values:
  1 - Home purchase
  2 - Home improvement
  31 - Refinancing
  32 - Cash-out refinancing
  4 - Other purpose
  5 - Not applicable
  ```
  +  lien_status -- Lien status of the property securing the covered loan, or in the case of an application, proposed to secure the covered loan
  ```

  ```
  + construction_method -- Construction method for the dwelling
  ```

  ```
  + occupancy_type -- Occupancy type for the dwelling
  ```

  ```
  + manufactured_home_secured_property_type -- Whether the covered loan or application is, or would have been, secured by a manufactured home and land, or by a manufactured home and not land
  ```

  ```
  + manufactured_home_land_property_interest -- The applicant’s or borrower’s land property interest in the land on which a manufactured home is, or will be, located
  ```

  ```
  + applicant_credit_score_type -- The name and version of the credit scoring model used to generate the credit score, or scores, relied on in making the credit decision
  ```
  
  ```
  + co-applicant_credit_score_type -- The name and version of the credit scoring model used to generate the credit score, or scores, relied on in making the credit decision
  ```

  ```
  + applicant_ethnicity-1 -- Ethnicity of the applicant or borrower
  ```

  ```
  + co-applicant_ethnicity-1 -- Ethnicity of the first co-applicant or co-borrower
  ```

  ```
  + applicant_race-1 -- Race of the applicant or borrower
  ```
  ```
  + co-applicant_race-1 -- Race of the first co-applicant or co-borrower
  ```
  ```


