Loan Fraud Detection
This project focuses on detecting fraudulent activities in loan applications using machine learning techniques. The primary goal is to identify patterns and anomalies in customer data that may indicate fraudulent behavior.
Overview
The dataset used in this project includes various features such as age, income, credit card spending, and more. The loan approval status (personal_loan) is used as the target variable to classify whether a loan application is fraudulent or legitimate. Fraud detection is critical for financial institutions to minimize risks and losses.
Key Features
Target Variable: personal_loan (1 = Loan Approved, 0 = Loan Not Approved).
Features Analyzed:
Income levels, credit card spending, and mortgage values.
Binary indicators such as securities account, CD account, and online banking usage.
Demographic information like age, family size, and education level.
Machine Learning Models:
Logistic Regression
Random Forest
XGBoost (with hyperparameter tuning)
Methodology
Data Cleaning: Removed invalid entries (e.g., negative years of experience).
Feature Engineering: Converted categorical variables to dummy variables and scaled numerical features.
Class Imbalance Handling: Used SMOTETomek to balance the dataset.
Model Evaluation: Evaluated models using metrics such as accuracy, precision, recall, and F1-score.
Results
The XGBoost model achieved the highest accuracy and F1-score, making it the most effective model for detecting fraudulent loan applications.
How to Recreate the Environment
Clone the repository.
Create a virtual environment:
bash


python3 -m venv venv
source venv/bin/activate
Install dependencies:
bash


pip install -r requirements.txt
Run the Python scripts to train and evaluate the models.
