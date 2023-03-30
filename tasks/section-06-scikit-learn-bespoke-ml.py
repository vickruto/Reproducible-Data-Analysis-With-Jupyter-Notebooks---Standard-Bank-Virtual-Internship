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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import sklearn
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = ['section-05-data-preparation']
product = None

# %% tags=["soorgeon-unpickle"]
X_train = pickle.loads(Path(upstream['section-05-data-preparation']['X_train']).read_bytes())
X_val = pickle.loads(Path(upstream['section-05-data-preparation']['X_val']).read_bytes())
y_train = pickle.loads(Path(upstream['section-05-data-preparation']['y_train']).read_bytes())
y_val = pickle.loads(Path(upstream['section-05-data-preparation']['y_val']).read_bytes())

# %% [markdown] id="5pfvdlsqGZT9"
# ## 06) Scikit-learn Bespoke ML

# %% id="DsieitWyPz-f" outputId="e7071457-f78e-4651-c89b-fb59baa19b51" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005836366, "user_tz": -180, "elapsed": 1706, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", accuracy_score(xgb.predict(X_train), y_train))
print("Accuracy on the Train Dataset : ", accuracy_score(xgb.predict(X_val), y_val))
print("\nConfusion matrix : \n", confusion_matrix(xgb.predict(X_val), y_val))

# %% id="CXU9jbpaLUq2"
# !pip install catboost
clear_output()

# %% id="ZK9BL9RbPz-g" outputId="d4078f2e-38c5-4225-d987-b479f0d90868" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005846505, "user_tz": -180, "elapsed": 4122, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}

cbc = CatBoostClassifier(verbose=0)
cbc.fit(X_train, y_train)

print("Accuracy on the Train Dataset : ", accuracy_score(cbc.predict(X_train), y_train))
print("Accuracy on the Train Dataset : ", accuracy_score(cbc.predict(X_val), y_val))
print("\nConfusion matrix : \n", confusion_matrix(cbc.predict(X_val), y_val))

# %% id="NwtHeGtkPz-g" outputId="f3cf9a39-f2c0-4436-9b26-b091fb8dd6eb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005847168, "user_tz": -180, "elapsed": 674, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", accuracy_score(lgbm.predict(X_train), y_train))
print("Accuracy on the Train Dataset : ", accuracy_score(lgbm.predict(X_val), y_val))
print("\nConfusion matrix : \n", confusion_matrix(lgbm.predict(X_val), y_val))

# %% id="kUrCfRIuLP0c"
## Some sklearn classifiers we can try

try:


  classifiers = sklearn.utils.all_estimators(type_filter='classifier')
  for name, class_ in classifiers:
      if hasattr(class_, 'predict_proba'):
          print(name, class_)
  
except:
  pass

# %% id="IwO2lqAvPz-h"
## Scale the data to a range of [0-1]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# %% id="d4XD_9K4E4Ue" outputId="182d1225-2a4c-4f82-c307-c99df0366a46" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005847812, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}

lr = LogisticRegression() 
lr.fit(X_train, y_train)

print("Accuracy on the Train Dataset : ", accuracy_score(lr.predict(X_train), y_train))
print("Accuracy on the Validation Dataset : ", accuracy_score(lr.predict(X_val), y_val))
print("\nConfusion matrix : \n", confusion_matrix(lr.predict(X_val), y_val))

# %% id="u2ZpGCLdPz-i" outputId="42e6ff94-ca44-4d16-e46c-6d77d012e9e6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005847813, "user_tz": -180, "elapsed": 19, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}

svm = SVC()
svm.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", svm.score(X_train, y_train))
print("Accuracy on the Validation Dataset : ", svm.score(X_val, y_val)) 
print("\nConfusion matrix : \n", confusion_matrix(svm.predict(X_val), y_val))

# %% id="tLvLaVYbPz-i" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005847813, "user_tz": -180, "elapsed": 14, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="4a6456fd-13dc-42bc-f5b4-ddcaebd036fd"

adb = AdaBoostClassifier()
adb.fit(X_train, y_train)
adb.score(X_train, y_train), adb.score(X_val, y_val)
print("Accuracy on the Train Dataset : ", adb.score(X_train, y_train))
print("Accuracy on the Validation Dataset : ", adb.score(X_val, y_val)) 
print("\nConfusion matrix : \n", confusion_matrix(adb.predict(X_val), y_val))

# %% id="IB0ZsuJdPz-i" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005847814, "user_tz": -180, "elapsed": 12, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="724764bc-1e0f-4503-81bb-513c694e69aa"

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", dt.score(X_train, y_train))
print("Accuracy on the Validation Dataset : ", dt.score(X_val, y_val)) 
print("\nConfusion matrix : \n", confusion_matrix(dt.predict(X_val), y_val))

# %% [markdown] id="rcteJlGYX1EE"
# The AutoML model has an accuracy of 78.9%. The bespoke classifier models tried have accuracies ranging from 71% to 80%. The models are tried right out of the box, which means that with parameter fine tuning, we can expect to get higher accuracies. Thus, we can make a case for bespoke modelling since accuracy is very important in this application case and any slight improvement in the expected level of accuracy should be pursued.

# %% tags=["soorgeon-pickle"]
Path(product['X_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_train']).write_bytes(pickle.dumps(X_train))

Path(product['X_val']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_val']).write_bytes(pickle.dumps(X_val))
