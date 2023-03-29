# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
# ---

# %% tags=["soorgeon-imports"]
from IPython.display import clear_output
import pandas as pd
import os
from google.colab import drive

## Import the data loading function
from jupyterworkflow.data import get_loan_data
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = None
product = None

# %% [markdown] id="A4eh9B5BWWRd"
# ## 01) Get data
# ### Install and  Import Libraries
#

# %% id="AsyeROA68RFh"

# !pip install sweetviz 
clear_output()
# !pip install auto-sklearn
clear_output()
# !pip install --upgrade scipy
clear_output()

# %% id="3FMXdjSw73HC"




 






 
# %matplotlib inline

# %% id="U0sg9yqHH5Io"
## There is a recurring error with autosklearn during its import 
## Retrying the import twice works 
for i in range(2):
  try:
    ## Try import autosklearn
    pass
  except:
    pass

# %% colab={"base_uri": "https://localhost:8080/"} id="yxlEjah5QPBN" executionInfo={"status": "ok", "timestamp": 1680005762187, "user_tz": -180, "elapsed": 139991, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="06da1261-1bbe-44b0-907c-cd03b0483d43"

drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="Q4dCSbUhAwut" executionInfo={"status": "ok", "timestamp": 1680005762744, "user_tz": -180, "elapsed": 561, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="b53158c8-32c7-43ec-af73-56c1252a2e35"
# %cd /content/drive/MyDrive/Colab Notebooks/Standard Bank Virtual Internship/

# %% [markdown] id="kU8I9U4KW9mz"
# ### Load Datasets

# %% colab={"base_uri": "https://localhost:8080/"} id="muKJod0rCi31" executionInfo={"status": "ok", "timestamp": 1680005762745, "user_tz": -180, "elapsed": 8, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="629bf960-b362-443a-abf2-cc556b4a6d10"
base_path = os.getcwd()
print(base_path)

# %% id="IoB9l70xZS4y"
kaggle_dataset = 'altruistdelhite04/loan-prediction-problem-dataset'
kaggle_credentials = '/content/drive/MyDrive/Colab Notebooks/kaggle.json'

train, test = get_loan_data(base_path, kaggle_dataset, kaggle_credentials)

# %% [markdown] id="b9aS4gz5ZKsE"
# # EDA

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="DfFhyw-xZPRN" outputId="3d283015-dfde-4213-9a8e-6a22f997fc02" executionInfo={"status": "ok", "timestamp": 1680005764719, "user_tz": -180, "elapsed": 33, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
train.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 287} id="o-4cXyj2ZVls" outputId="50dca8db-260e-475b-8793-4db9fd99a914" executionInfo={"status": "ok", "timestamp": 1680005764720, "user_tz": -180, "elapsed": 29, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
test.head()

# %% id="YvAVevIiZYw_" colab={"base_uri": "https://localhost:8080/", "height": 287} outputId="6b004bc7-d6a9-4e70-cdbc-51c3126b8505" executionInfo={"status": "ok", "timestamp": 1680005764721, "user_tz": -180, "elapsed": 29, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
# we concat for easy analysis
n = train.shape[0] # we set this to be able to separate the
df = pd.concat([train, test], axis=0)
df.head()

# %% tags=["soorgeon-pickle"]
Path(product['df']).parent.mkdir(exist_ok=True, parents=True)
Path(product['df']).write_bytes(pickle.dumps(df))

Path(product['train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['train']).write_bytes(pickle.dumps(train))
