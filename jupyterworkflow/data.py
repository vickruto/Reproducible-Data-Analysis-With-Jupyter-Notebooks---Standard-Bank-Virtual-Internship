import pandas as pd
import shutil
import os


def get_loan_data(base_path, kaggle_dataset=None, kaggle_credentials=None):
  '''
  Downloads dataset files from Kaggle if they have not be downloaded already and loads them into pd dataframes

  parameters 
  --------------- 
  base_path (string/os.path) -  the file path that contains the dataset or the path you want to download to  
  kaggle_dataset (string - optional) - the name of the dataset on kaggle. uses the format [kaggle_user]/[dataset_name]
  kaggle_credentials (string - optional) - the json file that contains the credentials to be used to access kaggle

  returns 
  --------------- 
  train (pd dataframe) - the loan data training set
  test (pd dataframe) - the loan data test set

  '''
  try : 
    train = pd.read_csv(os.path.join(base_path, 'train_u6lujuX_CVtuZ9i.csv'))
    test = pd.read_csv(os.path.join(base_path, 'test_Y3wMUE5_7gLdaTN.csv'))
  except FileNotFoundError: 
    if kaggle_dataset==None or kaggle_credentials==None:
      raise Exception("Dataset Files Not Found!\n\
        You may rerun the function, including the right dataset name and credentials to download them from Kaggle")


    ## Create a Kaggle API
    if not os.path.exists('/root/.kaggle'):
      os.makedirs('/root/.kaggle')
      shutil.copy(kaggle_credentials, '/root/.kaggle/kaggle.json')
      os.chmod('/root/.kaggle/kaggle.json', int('600', base=8))
      from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    ## Download missing datasets
    dataset_files = api.dataset_list_files(kaggle_dataset).files
    if not os.path.exists(base_path):
      os.makedirs(base_path)
    for file_ in dataset_files : 
      if str(file_) not in os.listdir(base_path):
        api.dataset_download_file(kaggle_dataset, str(file_), path=base_path)
    train = pd.read_csv(os.path.join(base_path, 'train_u6lujuX_CVtuZ9i.csv'))
    test = pd.read_csv(os.path.join(base_path, 'test_Y3wMUE5_7gLdaTN.csv'))
  print(f'Found {train.shape[0]} records in the train set with {train.shape[1]} columns')
  print(f'Found {test.shape[0]} records in the test set with {test.shape[1]} columns')
  return train, test
