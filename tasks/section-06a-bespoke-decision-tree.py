
# %% tags=["soorgeon-imports"]
from sklearn.metrics import confusion_matrix
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


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print("Accuracy on the Train Dataset : ", dt.score(X_train, y_train))
print("Accuracy on the Validation Dataset : ", dt.score(X_val, y_val)) 
print("\nConfusion matrix : \n", confusion_matrix(dt.predict(X_val), y_val))

