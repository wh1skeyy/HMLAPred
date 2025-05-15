# Predict House Mortgage Loan Approval
## Project Overview
For the chosen dataset of this project, we visited [ffiec.cfpb.gov](https://ffiec.cfpb.gov/data-browser/data/2023?category=states) (which is the U.S Home Mortgage Disclosure Act website) and filtered on the public records to suit the need of later analysis.

For better understand of the project's scope, the followings are the filters input:

*(Disclaimer: the data file is quite big -- 2,9 GB)*

Filter 1: Geography -- Nationwide

Filter 2: Financial Institution -- All Financial Institution

Filter 3: Popular variables -- Action taken

(Once chosen 'Action taken', there are additional filters. And the selected ones are **Loan Originated, Application approved but not accepted, Application denied, Preapproval request denied, Preapproval request approved but not accepted** )

[Link to the dataset]()
*(Disclaimer: since the dataset is quite big -- 2,9 GB, it cannot be uploaded onto Github. And so we uploaded it to drive instead. The cleaned data is also in the same folder.)*

**PROJECT PURPOSE**: Using the dataset, we aim to train a model that can **predict the loan approaval for a House Mortgage**. With a little attention on the 'Action taken' filters, you can see that those are the possible prediction results we are to generate. Among which, 'Loan Originated' and 'Application approved but not accepted' are approved loans; the rest are loans that get denied.

### Introduction to the dataset
The dataset is of raw format, consisting of 100 columns and over 8.2 million observations (each column recored a piece of informatiom on the loaner)

*For more details on the variables, please open 'var_des.txt'*

## Project Details
### Data Cleaning and Processing

* Import data
```
import pandas as pd
data = pd.read_csv('data.csv')
```

* Label Encoding
```
# Encoding 'action_taken' -- an important variable
print(df['action_taken'].unique())
action_taken_mapping = {
    1: "Loan originated",
    2: "Application approved but not accepted",
    3: "Application denied",
    7: "Preapproval request denied",
    8: "Preapproval request approved but not accepted"
}

df['action_taken'] = df['action_taken'].map(action_taken_mapping)
```

* Remove unnecessary columns
```
columns_to_drop = [
    'lei',
    'derived_msa-md',
    'state_code',
    'county_code',
    .....
    'tract_median_age_of_housing_units',
    'multifamily_affordable_units'
]
data = data.drop(columns=columns_to_drop)
```

* Fill the remaining variables with 'Unknown' or similar phrase
```
data['co-applicant_age_above_62'] = data['co-applicant_age_above_62'].fillna("No Co-Applicant")
data['applicant_age_above_62'] = data['applicant_age_above_62'].fillna("Unknown")
```

* Drop rows that have NAs in specific columns (as followed):
```
columns_to_check = ['applicant_ethnicity-1', 'co-applicant_ethnicity-1', 'applicant_race-1', 'co-applicant_race-1']
data = data.dropna(subset=columns_to_check)
```
*Explantion:* These are personal details that cannot be replaced. Plus, dropping would not significantly affect the model later

#### Handling important financial variables 
* Transfer the variables to numeric
```
import numpy as np

numeric_cols = ['income', 'loan_term', 'loan_to_value_ratio', 'debt_to_income_ratio', 'interest_rate', 'rate_spread', 'total_loan_costs', 'total_points_and_fees', 'loan_term', 'origination_charges', 'discount_points']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```
* Change 'exempt' or 'NaN' to 'Unknown'
```
for col in ['income', 'loan_term', 'loan_to_value_ratio', 'debt_to_income_ratio']:
    df[col] = df[col].replace('exempt', 'Unknown')


for col in ['interest_rate', 'rate_spread', 'total_loan_costs', 'total_points_and_fees', 'loan_term', 'origination_charges', 'discount_points']:
    df[col] = df[col].astype(str).replace(['exempt', 'nan'], 'Unknown')

```
* Classify the values of each variables
```
# Example of 'loan_term'
def categorize_loan_term(term):
    if term == 'Missing' or pd.isna(term):
        return 'Missing'
    try:
        term = float(term)
        if term <= 180:
            return 'Short'
        elif term <= 240:
            return 'Medium'
        else:
            return 'Long'
    except:
        return 'Missing'

df['loan_term_category'] = df['loan_term'].apply(categorize_loan_term)
```
* Assign values based on action taken
```
# Example of total loan costs

df['total_loan_costs'] = df.apply(
    lambda row: 0 if pd.isna(row['total_loan_costs']) and row['action_taken'] == 3 else row['total_loan_costs'], axis=1)
df['total_loan_costs'] = df['total_loan_costs'].fillna(df['total_loan_costs'].median())
```
*Explanation*: 2 conditions are checked for every observation, whether the value in the 'total_loan_cost' is missing and if the 'action taken' is 3 -- loan denied. If both conditions are true, the value '0' shall be assigned to 'total_loan_cost', otherwise, it is unchanged. However, following this, the rows with missing 'total_loan_cost' get the column's median value.

* Save to csv for later analysis
```
df.to_csv('data_fixed.csv', index=False)
```

## EDA (PySpark is used in this session)
* Import library and set up environment
```
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, count, approx_count_distinct, when
from pyspark.sql.types import IntegerType, DoubleType
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import col, mean, count, when

# Set JAVA_HOME environment variable (update the path to your Java installation)
os.environ["JAVA_HOME"] = "C:\\Program Files\\OpenLogic\\jdk-11.0.26.4-hotspot"  # Replace with your Java path
os.environ["PATH"] = os.environ["JAVA_HOME"] + "\\bin;" + os.environ["PATH"]
```
* Import cleaned data
```
# Initialize Spark Session
spark = SparkSession.builder \
    .appName("HMDA EDA with PySpark") \
    .getOrCreate()

df = spark.read.csv("data_fixed_3.csv", header=True, inferSchema=True)
df = df.sample(withReplacement=False, fraction=0.15, seed=42)
```
### Credit variables' general distribution
```
# Credit variables
credit_vars = ['loan_amount', 'interest_rate', 'debt_to_income_ratio', 'income','ltv_category', 'applicant_credit_score_type', 'applicant_age']

# Tranfer to pandas for visualization
pd_df = df.select(credit_vars).toPandas()

# Create subplot 
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Interest Rate Distribution
sns.histplot(data=pd_df, x='interest_rate', kde=True, ax=axes[0])
axes[0].set_title('Distribution of Interest Rate')

# DTI Distribution
sns.histplot(data=pd_df, x='debt_to_income_ratio', kde=True, ax=axes[1])
axes[1].set_title('Distribution of Debt-to-Income Ratio')

# LTV Categories Distribution
sns.countplot(data=pd_df, x='ltv_category', ax=axes[2])
axes[2].set_title('Distribution of LTV Categories')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```
![i1](https://i.imgur.com/u24m9Fj.png)
![i2](https://i.imgur.com/5QayXhZ.png)
![i3](https://i.imgur.com/j7wMnm2.png)

*Insight:*
- The number of loans with an interest rate of 15% is distinguishly larger than the rest. This brings a speculation that majority of the loans are of high risk.
-  The density of debt-to-income-ratio peaks at 42%, 44%, and 48% -- which are considered high for this ratio. This once again suggests that the concentrations of loans are on the 'high risk' side.
- Though the largest portion of the loans are of small value, the number of those with high value is quite considerable.

### Variables' Distribution by risk group
![i10](https://i.imgur.com/3Wr7YIU.png)

*Among the financial variables that we picked out, the one that usually has the nost influence in a loan approval decision is 'debt_to_income_ratio'. Hence, we created a 'risk group' column in the dataframe, where values are assigned to rows based on 'debt_to_income_ratio'. Specifically, observations with debt_to_income_ratio equal or below 36 are considered 'low risk'; ones with the ratio above 43 are 'high risk'; and the rest are 'medium risk'.*
```
df = df.withColumn('risk_group',
    when(col('debt_to_income_ratio') <= 36, 'Low Risk')
    .when((col('debt_to_income_ratio') > 36) & (col('debt_to_income_ratio') <= 43), 'Medium Risk')
    .otherwise('High Risk'))
```

Following that, a statistic table is created, consisting of the mean of 'loan_amount' & 'interest rate' along with the most common 'loan_total_value_category' at each risk level.
```
risk_stats = df.groupBy('risk_group').agg(
    count('*').alias('count'),
    mean('loan_amount').alias('avg_loan_amount'),
    mean('interest_rate').alias('avg_interest_rate'),
    F.mode('ltv_category').alias('most_common_ltv_category')
)
risk_stats.show()
```
![i4](https://i.imgur.com/lFnJrxC.png)
*Insight:*
- As confirmation to the previous speculation, it has indeed been proven that the loans of high risk are substantially bigger in number compared to loans of low and medium risk. 
- The average loan amount and interest rate of high risk loans are also the highest (while the interest rate of the other two are close in nunber). After all, this is to be expected, as realistically, an  high-risk borrower applies for a loan, a high interest rate is to be expected.

Afterward, the distribution of each variable in the table at different risk level is visualized.
```
# Tranfer to pandas for visualization
pd_risk = df.select('risk_group', 'loan_amount', 'interest_rate', 'ltv_category').toPandas()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Loan Amount by Risk Group
sns.boxplot(data=pd_risk, x='risk_group', y='loan_amount', ax=axes[0])
axes[0].set_title('Loan Amount by Risk Group')

# Interest Rate by Risk Group
sns.boxplot(data=pd_risk, x='risk_group', y='interest_rate', ax=axes[1])
axes[1].set_title('Interest Rate by Risk Group')

# LTV Categories by Risk Group
ltv_risk_crosstab = pd.crosstab(pd_risk['risk_group'], pd_risk['ltv_category'], normalize='index')
ltv_risk_crosstab.plot(kind='bar', stacked=True, ax=axes[2])
axes[2].set_title('LTV Categories Distribution by Risk Group')
axes[2].legend(title='LTV Category', bbox_to_anchor=(1.05, 1))
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```
![i5](https://i.imgur.com/6efB45m.png)
![i6](https://i.imgur.com/Uy6WHTS.png)
![i7](https://i.imgur.com/t6F5K9Q.png)
*Insight:*
- High Risk loans has more variability and outliers, with some loans exceeding $1 million. Seemingly, high-risk borrowers tend to borrow larger amounts than the other. This mmust have been a further push for the higher interest rate appearent in this specific group.  
- For details on interest rate, High Risk loans has the highest median (~10%), Medium Risk (~7.5%), and Low Risk (~5%). Among them, high risk shows the most variability, with outliers up to 17.5%.
- This has been shown in the previous table, but regardless of risk level, the total value of the majority of loans is low.
## ML Model 
*Description: In this part, we used the data set to train different types of ML model, then evaluate their predictions to choose the most prescise one for the web prediction application later on*

* Import Libraries
```
%pip install catboost lightgbm
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```

* Encode categorical columns
```
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col].astype(str))
```



* Split train/ test files
```
X = df.drop(columns=['action_taken'])  # Dropping non-relevant columns
Y = df['action_taken']  # Target variable

x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2 , random_state=69 , shuffle=True)
```
*(The test files are used to examined the models' predictions -- which would later be compared to the real data for accuracy. This will decide which model is chosen for the prediction application)*

* Initialize and fit the classifiers
```

# Initialize classifiers with moderate complexity
cls1 = GradientBoostingClassifier(n_estimators=10)
cls2 = LGBMClassifier(n_estimators=10)
cls3 = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
cls4 = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=10)
cls5 = XGBClassifier(n_estimators=10, max_depth=6, eval_metric='logloss')
cls6 = RandomForestClassifier(n_estimators=10)
cls7 = ExtraTreesClassifier(n_estimators=10)
cls8 = DecisionTreeClassifier()
cls9 = CatBoostClassifier(logging_level='Silent', n_estimators=10)

# Fit the classifiers
cls1.fit(x_train, y_train)
cls2.fit(x_train, y_train)
....
cls9.fit(x_train, y_train)
```
* Prediction results
```
pred1 = cls1.predict(x_test)
pred2 = cls2.predict(x_test)
...
pred9 = cls9.predict(x_test)
```

* Evaluation
```
# Calculate accuracy for each classifier
acc_values = [
    accuracy_score(y_test, pred1),
    accuracy_score(y_test, pred2),
    accuracy_score(y_test, pred3),
    accuracy_score(y_test, pred4),
    accuracy_score(y_test, pred5),
    accuracy_score(y_test, pred6),
    accuracy_score(y_test, pred7),
    accuracy_score(y_test, pred8),
    accuracy_score(y_test, pred9)
]

# Sort accuracy values in descending order (high to low)
acc_sorted_indices = np.argsort(acc_values)[::-1]
acc_values_sorted = np.array(acc_values)[acc_sorted_indices]
classifier_names_acc_sorted = np.array(classifier_names)[acc_sorted_indices]
```
![i8](https://i.imgur.com/ACeh7O1.png)
*Insight*: As the result shows, the accuracy of the models is very close to each other, and it does not help much in choosing a model. And so, we went for a deeper inspection, which is extracting feature importances of top 3 most accurate models, namely, Bagging Classifier, Random Forest and LGBM.
* Accuracy Visualization
```
plt.barh(classifier_names_acc_sorted, acc_values_sorted, color='skyblue')
plt.title('Accuracy of Classifiers (High to Low)')
plt.xlabel('Accuracy')
plt.xlim(0, 1)  # Assuming accuracy ranges from 0 to 1
plt.gca().invert_yaxis()  # Best performance on top

plt.tight_layout()
plt.show()
```
![i9](https://i.imgur.com/HnqPfpJ.png)
*Insight*: According to the final results, LGBM Classifier considers a great number of input features when making predictions, thus, seems less biased than the other 2 models. Therefore, we chose LGBM as our prediction model for the web application.
