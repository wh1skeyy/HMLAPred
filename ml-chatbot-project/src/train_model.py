import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv('data-2.csv')

# Preprocess the data
df['ApplicationDate'] = pd.to_datetime(df['ApplicationDate'])
encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

# Define features and target variable
X = df.drop(columns=['ApplicationDate', 'RiskScore'])
y = df['RiskScore']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/risk_score_model.pkl')