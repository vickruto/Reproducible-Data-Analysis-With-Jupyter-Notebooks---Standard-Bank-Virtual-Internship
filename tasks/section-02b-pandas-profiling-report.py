# %% tags=["soorgeon-imports"]
from ydata_profiling import ProfileReport
from pathlib import Path
import pickle
import os 

# %% tags=["parameters"]
upstream = ['section-01-get-data']
product = None

# %% tags=["soorgeon-unpickle"]
train = pickle.loads(Path(upstream['section-01-get-data']['train']).read_bytes())

# %%
if not os.path.exists('reports/auto-generated/'):
  os.makedirs('reports/auto-generated/')
print(os.getcwd())

# %% 
profile = ProfileReport(train, title="Profiling Report")
profile.to_file("reports/auto-generated/PANDAS_PROFILING_REPORT.html")
