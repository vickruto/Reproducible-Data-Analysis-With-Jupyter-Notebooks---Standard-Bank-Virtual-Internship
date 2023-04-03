from pathlib import Path
import pickle

def test_loan_data(product):
  train = pickle.loads(Path(product['train']).read_bytes())
  df = pickle.loads(Path(product['df']).read_bytes())

  train_columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education','Self_Employed', 
                  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                  'Credit_History', 'Property_Area', 'Loan_Status']

  ## The train dataset columns is a superset of the test dataset columns
  ## Concatanating should result in the train columns 
  assert all(df.columns == train_columns)

  ## Expected number of records in the train dataset
  assert train.shape[0] == 614

  ## Number of nulls of the target variable in the concatenated df should be the 
  ##  same as the expected number of records in the test set
  assert df['Loan_Status'].isnull().sum() == 367
  assert df.shape[0] == 614+367

  ## Assert there are not duplicates
  assert df.duplicated().sum() == 0

