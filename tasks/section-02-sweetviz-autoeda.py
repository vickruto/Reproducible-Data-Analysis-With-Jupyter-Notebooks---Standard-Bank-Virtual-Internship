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
import sweetviz
from IPython.display import display, HTML
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = ['section-01-get-data']
product = None

# %% tags=["soorgeon-unpickle"]
train = pickle.loads(Path(upstream['section-01-get-data']['train']).read_bytes())

# %% [markdown] id="cdTHtXlKt3_k"
# ## 02) Sweetviz AutoEDA

# %% colab={"base_uri": "https://localhost:8080/", "height": 803, "referenced_widgets": ["9ff884501ee44bb3a42d83e84761ce9b", "0aae9ce35c2a40e8b27c04d8337cf8ac", "1266a06f94964a4d9aaf968c415fbebe", "85e0bd4c4ee64ae690163efddef12ab4", "81880667571e44399b5d27889bf8af43", "2f39ce49dfc748a6a04eaea4fa0a9183", "8f345a5ce61a4650b3eb19e714965148", "23adcd83489f4b0b8402ecb351b2f9b6", "555b74933e9243329ea047a1f241ba98", "7e0b3ed237e44383a6b4bfc8988f61bb", "bdd912dec483473793b6d2a5d3a24a37"]} id="fAKTNiGXuCRf" outputId="b23f495f-bb0d-4314-b518-1b82b024d854" executionInfo={"status": "ok", "timestamp": 1680005770809, "user_tz": -180, "elapsed": 6116, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
autoEDA = sweetviz.analyze(train)
#autoEDA.show_html()
autoEDA.show_notebook()

# %% id="974YfMypFNJm" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1680005889001, "user_tz": -180, "elapsed": 2509, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}} outputId="10d73794-1504-4803-c11c-e44b8edc238d"

display(HTML('SWEETVIZ_REPORT.html'))
