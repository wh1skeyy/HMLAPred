from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load the trained model (assuming it's saved as a pickle file)
model = joblib.load('/workspaces/py-final/risk_score_model.pkl')

# Dummy model for demonstration purposes
class DummyModel:
    def predict(self, input_data):
        return [0.5]  # Replace with actual model prediction logic

model = DummyModel()

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_risk_score = None
    if request.method == 'POST':
        # Get input data from the form
        input_data = {
            'Age': int(request.form['Age']),
            'AnnualIncome': int(request.form['AnnualIncome']),
            'CreditScore': int(request.form['CreditScore']),
            'EmploymentStatus': int(request.form['EmploymentStatus']),
            'EducationLevel': int(request.form['EducationLevel']),
            'Experience': int(request.form['Experience']),
            'LoanAmount': int(request.form['LoanAmount']),
            'LoanDuration': int(request.form['LoanDuration']),
            'MaritalStatus': int(request.form['MaritalStatus']),
            'NumberOfDependents': int(request.form['NumberOfDependents']),
            'HomeOwnershipStatus': int(request.form['HomeOwnershipStatus']),
            'MonthlyDebtPayments': float(request.form['MonthlyDebtPayments']),
            'CreditCardUtilizationRate': float(request.form['CreditCardUtilizationRate']),
            'NumberOfOpenCreditLines': int(request.form['NumberOfOpenCreditLines']),
            'NumberOfCreditInquiries': int(request.form['NumberOfCreditInquiries']),
            'DebtToIncomeRatio': float(request.form['DebtToIncomeRatio']),
            'BankruptcyHistory': int(request.form['BankruptcyHistory']),
            'LoanPurpose': int(request.form['LoanPurpose']),
            'PreviousLoanDefaults': int(request.form['PreviousLoanDefaults']),
            'PaymentHistory': int(request.form['PaymentHistory']),
            'LengthOfCreditHistory': int(request.form['LengthOfCreditHistory']),
            'SavingsAccountBalance': int(request.form['SavingsAccountBalance']),
            'CheckingAccountBalance': int(request.form['CheckingAccountBalance']),
            'TotalAssets': int(request.form['TotalAssets']),
            'TotalLiabilities': int(request.form['TotalLiabilities']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'UtilityBillsPaymentHistory': float(request.form['UtilityBillsPaymentHistory']),
            'JobTenure': int(request.form['JobTenure']),
            'NetWorth': int(request.form['NetWorth']),
            'BaseInterestRate': float(request.form['BaseInterestRate']),
            'InterestRate': float(request.form['InterestRate']),
            'MonthlyLoanPayment': float(request.form['MonthlyLoanPayment']),
            'TotalDebtToIncomeRatio': float(request.form['TotalDebtToIncomeRatio']),
        }


        # Create a DataFrame for the input data
        input_df = pd.DataFrame([input_data])

        # Use the model to predict the risk score
        predicted_risk_score = model.predict(input_df)[0]

    return render_template('form.html', predicted_risk_score=predicted_risk_score)

if __name__ == '__main__':
    app.run(debug=True)