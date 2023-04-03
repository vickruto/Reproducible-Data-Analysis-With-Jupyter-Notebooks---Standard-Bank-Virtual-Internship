# %% tags=["soorgeon-imports"]
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler

# %% tags=["parameters"]
upstream = ['section-05-data-preparation']
product = None

# %% tags=["soorgeon-unpickle"]
X_train = pickle.loads(Path(upstream['section-05-data-preparation']['X_train']).read_bytes())
X_val = pickle.loads(Path(upstream['section-05-data-preparation']['X_val']).read_bytes())
y_train = pickle.loads(Path(upstream['section-05-data-preparation']['y_train']).read_bytes())
y_val = pickle.loads(Path(upstream['section-05-data-preparation']['y_val']).read_bytes())

# %%
## Standardize features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# %% tags=["soorgeon-pickle"]
Path(product['X_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_train']).write_bytes(pickle.dumps(X_train))

Path(product['X_val']).parent.mkdir(exist_ok=True, parents=True)
Path(product['X_val']).write_bytes(pickle.dumps(X_val))

Path(product['y_train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['y_train']).write_bytes(pickle.dumps(y_train))

Path(product['y_val']).parent.mkdir(exist_ok=True, parents=True)
Path(product['y_val']).write_bytes(pickle.dumps(y_val))
