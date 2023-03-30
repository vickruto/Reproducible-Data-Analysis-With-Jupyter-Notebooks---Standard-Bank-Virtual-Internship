
# %% tags=["soorgeon-imports"]
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
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

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", accuracy_score(xgb.predict(X_train), y_train))
print("Accuracy on the Train Dataset : ", accuracy_score(xgb.predict(X_val), y_val))
print("\nConfusion matrix : \n", confusion_matrix(xgb.predict(X_val), y_val))
