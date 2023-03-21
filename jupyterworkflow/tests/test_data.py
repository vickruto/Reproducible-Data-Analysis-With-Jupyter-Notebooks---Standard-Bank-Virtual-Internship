import pandas as pd
from jupyterworkflow.data import get_loan_data

def test_get_loan_data(base_path='.', kaggle_dataset=None, kaggle_credentials=None):
  train, test = get_loan_data(base_path, kaggle_dataset, kaggle_credentials)
  train_columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 
                  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                  'Credit_History', 'Property_Area', 'Loan_Status']

  test_columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
                  'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                  'Loan_Amount_Term', 'Credit_History', 'Property_Area']

  assert all(train.columns == train_columns)
  assert all(test.columns == test_columns)
  assert train.shape[0] == 614
  assert test.shape[0] == 367

