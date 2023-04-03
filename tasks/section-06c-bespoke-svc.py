
# %% tags=["soorgeon-imports"]
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = ['section-05b-standardize-features']
product = None

# %% tags=["soorgeon-unpickle"]
X_train = pickle.loads(Path(upstream['section-05b-standardize-features']['X_train']).read_bytes())
X_val = pickle.loads(Path(upstream['section-05b-standardize-features']['X_val']).read_bytes())
y_train = pickle.loads(Path(upstream['section-05b-standardize-features']['y_train']).read_bytes())
y_val = pickle.loads(Path(upstream['section-05b-standardize-features']['y_val']).read_bytes())

# %%
svm = SVC()
svm.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", svm.score(X_train, y_train))
print("Accuracy on the Validation Dataset : ", svm.score(X_val, y_val)) 
print("\nConfusion matrix : \n", confusion_matrix(svm.predict(X_val), y_val))
