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
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = ['section-01-get-data', 'section-04-autosklearn-auto-ml']
product = None

# %% tags=["soorgeon-unpickle"]
train = pickle.loads(Path(upstream['section-01-get-data']['train']).read_bytes())
X = pickle.loads(Path(upstream['section-04-autosklearn-auto-ml']['X']).read_bytes())
categorical_columns = pickle.loads(Path(upstream['section-04-autosklearn-auto-ml']['categorical_columns']).read_bytes())
y = pickle.loads(Path(upstream['section-04-autosklearn-auto-ml']['y']).read_bytes())

# %% [markdown] id="QKCOAsTS-J1P"
# ## 05) Data Preparation

# %% colab={"base_uri": "https://localhost:8080/"} id="bXyiuKoTEanY" executionInfo={"status": "ok", "timestamp": 1680005834679, "user_tz": -180, "elapsed": 24, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="f4445c36-f60e-434d-912d-59bb4dd24efc"
## One hot encode the categorical features
##Leave all categories to represent missing values 
X_oh_encoded = pd.concat([X, pd.get_dummies(X[categorical_columns])], axis=1)
X_oh_encoded = X_oh_encoded.drop(columns=categorical_columns)

## fill the missing numerical columns
X_oh_encoded['LoanAmount'].fillna(X_oh_encoded['LoanAmount'].mean(), inplace=True)
X_oh_encoded['Loan_Amount_Term'].fillna(X_oh_encoded['Loan_Amount_Term'].median(), inplace=True)
X_oh_encoded['Credit_History'].fillna(X_oh_encoded['Credit_History'].mode()[0], inplace=True)
X_oh_encoded.isna().sum()

# %% id="4EM0I7tfPz-f"
X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=['Loan_ID', 'Loan_Status']), train['Loan_Status'])

# %% id="nQw-9zvPK68R"
X_train, X_val, y_train, y_val = train_test_split(X_oh_encoded, y)

# %% tags=["soorgeon-pickle"]
Path(product['X_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_train']).write_bytes(pickle.dumps(X_train))

Path(product['X_val']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_val']).write_bytes(pickle.dumps(X_val))

Path(product['y_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['y_train']).write_bytes(pickle.dumps(y_train))

Path(product['y_val']).parent.mkdir(exist_ok=True, parents=True)
Path(product['y_val']).write_bytes(pickle.dumps(y_val))
