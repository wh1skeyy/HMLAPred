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
   Values:
   1 - Secured by a first lien
   2 - Secured by a subordinate lien
  ```
  + construction_method -- Construction method for the dwelling
  ```
  1 - Site-built
  2 - Manufactured home
  ```
  + occupancy_type -- Occupancy type for the dwelling
  ```
  1 - Principal residence
  2 - Second residence
  3 - Investment property
  ```
  + manufactured_home_secured_property_type -- Whether the covered loan or application is, or would have been, secured by a manufactured home and land, or by a manufactured home and not land
  ```
  1 - Manufactured home and land
  2 - Manufactured home and not land
  3 - Not applicable
  1111 - Exempt
  ```
  + manufactured_home_land_property_interest -- The applicant’s or borrower’s land property interest in the land on which a manufactured home is, or will be, located
  ```
  1 - Direct ownership
  2 - Indirect ownership
  3 - Paid leasehold
  4 - Unpaid leasehold
  5 - Not applicable
  1111 - Exempt
  ```
  + applicant_credit_score_type -- The name and version of the credit scoring model used to generate the credit score, or scores, relied on in making the credit decision
  ```
  1 - Equifax Beacon 5.0
  2 - Experian Fair Isaac
  3 - FICO Risk Score Classic 04
  4 - FICO Risk Score Classic 98
  5 - VantageScore 2.0
  6 - VantageScore 3.0
  7 - More than one credit scoring model
  8 - Other credit scoring model
  9 - Not applicable
  1111 - Exempt
  ```
  + co-applicant_credit_score_type -- The name and version of the credit scoring model used to generate the credit score, or scores, relied on in making the credit decision
  ```
  1 - Equifax Beacon 5.0
  2 - Experian Fair Isaac
  3 - FICO Risk Score Classic 04
  4 - FICO Risk Score Classic 98
  5 - VantageScore 2.0
  6 - VantageScore 3.0
  7 - More than one credit scoring model
  8 - Other credit scoring model
  9 - Not applicable
  10 - No co-applicant
  1111 - Exempt
  ```
  + applicant_ethnicity-1 -- Ethnicity of the applicant or borrower
  ```
  1 - Hispanic or Latino
  11 - Mexican
  12 - Puerto Rican
  13 - Cuban
  14 - Other Hispanic or Latino
  2 - Not Hispanic or Latino
  3 - Information not provided by applicant in mail, internet, or telephone application
  4 - Not applicable
  ```
  + co-applicant_ethnicity-1 -- Ethnicity of the first co-applicant or co-borrower
  ```
  similar to applicant_ethnicity-1
  5 - No co-applicant
  ```
  + applicant_race-1 -- Race of the applicant or borrower
  ```
  1 - American Indian or Alaska Native
  2 - Asian
  21 - Asian Indian
  22 - Chinese
  23 - Filipino
  24 - Japanese
  25 - Korean
  26 - Vietnamese
  27 - Other Asian
  3 - Black or African American
  4 - Native Hawaiian or Other Pacific Islander
  41 - Native Hawaiian
  42 - Guamanian or Chamorro
  43 - Samoan
  44 - Other Pacific Islander
  5 - White
  6 - Information not provided by applicant in mail, internet, or telephone application
  7 - Not applicable
  ```
  + co-applicant_race-1 -- Race of the first co-applicant or co-borrower
  ```
  similar to applicant_race-1
  8 - No co-applicant
  ```
  + applicant_age_above_62 --  Whether the applicant or borrower age is 62 or above
  ```
  0 - No
  1 - Yes
  ```
  + co-applicant_age_above_62
  ```
  0 - No
  1 - Yes
  ```
  + aus-1 -- The automated underwriting system(s) (AUS) used by the financial institution to evaluate the application
  ```
  1 - Desktop Underwriter (DU)
  2 - Loan Prospector (LP) or Loan Product Advisor
  3 - Technology Open to Approved Lenders (TOTAL) Scorecard
  4 - Guaranteed Underwriting System (GUS)
  5 - Other
  6 - Not applicable
  7 - Internal Proprietary System
  1111 - Exempt
  ```
  + denial_reason-1 -- The principal reason, or reasons, for denial
  ```
  1 - Debt-to-income ratio
  2 - Employment history
  3 - Credit history
  4 - Collateral
  5 - Insufficient cash (downpayment, closing costs)
  6 - Unverifiable information
  7 - Credit application incomplete
  8 - Mortgage insurance denied
  9 - Other
  10 - Not applicable
  ```
  + ltv_category -- classification of loan to value
  ```
  0 - High
  1 - Low
  2 - Medium
  3 - Missing
  ```
  + dti_category -- classification of debt-to-income ratio
  ```
  0 - 30-50
  1 - <30
  2 - >50
  3 - Missing
  ```
  + loan_term_category -- classification of loan term
  ```
  0 - Long
  1 - Medium
  2 - Missing
  3 - Short
  ```






