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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
for i in range(2):
  try:
    from autosklearn.classification import AutoSklearnClassifier
  except:
    pass
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = ['section-01-get-data']
product = None

# %% tags=["soorgeon-unpickle"]
train = pickle.loads(Path(upstream['section-01-get-data']['train']).read_bytes())

# %% [markdown] id="huNBju-TMV88"
# ## 04) Autosklearn  Auto ML
#

# %% colab={"base_uri": "https://localhost:8080/"} id="VMYiRQivBvS9" executionInfo={"status": "ok", "timestamp": 1680005770833, "user_tz": -180, "elapsed": 109, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="d147e8e8-3032-425b-9db2-f567aac91e81"
feature_columns = train.columns[1:-1]
print(feature_columns)

# %% colab={"base_uri": "https://localhost:8080/"} id="80HRr4R9COtP" executionInfo={"status": "ok", "timestamp": 1680005770834, "user_tz": -180, "elapsed": 102, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="75c3ce5e-3169-4611-8aba-a6b864ef772e"
## Input data with feature columns 
X = train[feature_columns].copy()

## Convert categorical features to 'category' type
categorical_columns = ['Gender', 'Married','Dependents', 'Education', 'Self_Employed', 'Property_Area']
X[categorical_columns] = X[categorical_columns].astype('category')
X.dtypes

# %% id="wDlAmmP6MgQF"
## label encode target
y = train['Loan_Status'].map({'N':0,'Y':1}).astype(int)

## train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% id="9wwYMdK_McJt"
# train
autoML = AutoSklearnClassifier(time_left_for_this_task=2*30, per_run_time_limit=30, n_jobs=8) # imposing a 1 minute time limit on this
autoML.fit(X_train, y_train)

# predict
predictions_autoML = autoML.predict(X_test)

# %% id="qnbGH9zS-WYh" colab={"base_uri": "https://localhost:8080/"} outputId="48c4b39e-c200-4c22-f0d4-0b75f1a8b463" executionInfo={"status": "ok", "timestamp": 1680005834679, "user_tz": -180, "elapsed": 36, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
print('autoML Model Accuracy:', accuracy_score(predictions_autoML, y_test))

# %% id="pNPPfGUtOnJ9" colab={"base_uri": "https://localhost:8080/"} outputId="7d72b84d-054e-4304-fc72-c39c9c6b1167" executionInfo={"status": "ok", "timestamp": 1680005834679, "user_tz": -180, "elapsed": 31, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
print(confusion_matrix(predictions_autoML, y_test))

# %% tags=["soorgeon-pickle"]
Path(product['X']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X']).write_bytes(pickle.dumps(X))

Path(product['X_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_train']).write_bytes(pickle.dumps(X_train))

Path(product['categorical_columns']).parent.mkdir(exist_ok=True, parents=True)
Path(product['categorical_columns']).write_bytes(pickle.dumps(categorical_columns))

Path(product['y']).parent.mkdir(exist_ok=True, parents=True)
Path(product['y']).write_bytes(pickle.dumps(y))

Path(product['y_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['y_train']).write_bytes(pickle.dumps(y_train))
