# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None
product = None

# -

import pandas as pd
from pathlib import Path
import pickle
import os
print(os.getcwd())
print()

train_url = 'https://raw.githubusercontent.com/vickruto/Reproducible-Data-Analysis-With-Jupyter-Notebooks-Standard-Bank-Virtual-Internship/ploomber-workflow/train_u6lujuX_CVtuZ9i.csv'
test_url = 'https://raw.githubusercontent.com/vickruto/Reproducible-Data-Analysis-With-Jupyter-Notebooks-Standard-Bank-Virtual-Internship/ploomber-workflow/test_Y3wMUE5_7gLdaTN.csv'
train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

df = pd.concat([train, test], axis=0)
df.head()

Path(product['df']).parent.mkdir(exist_ok=True, parents=True)
Path(product['df']).write_bytes(pickle.dumps(df))

Path(product['train']).parent.mkdir(exist_ok=True, parents=True)
Path(product['train']).write_bytes(pickle.dumps(train))
