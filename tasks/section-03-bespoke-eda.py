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
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from pathlib import Path
import pickle

# %% tags=["parameters"]
upstream = ['section-01-get-data']
product = None

# %% tags=["soorgeon-unpickle"]
df = pickle.loads(Path(upstream['section-01-get-data']['df']).read_bytes())
train = pickle.loads(Path(upstream['section-01-get-data']['train']).read_bytes())

# %% [markdown] id="f4xUxUnWw9lM"
# ## 03) Bespoke EDA

# %% id="suF8EwmEPz-W" outputId="fdfcb1ea-70fb-4ae3-f73d-50b681fe2732" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005770811, "user_tz": -180, "elapsed": 139, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Check for duplicates in the dataset
df.duplicated().sum()

# %% id="icnfcLWCxSVF" outputId="2da5df66-8722-4d86-eabd-335af32bdfff" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005770812, "user_tz": -180, "elapsed": 136, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Check for null values 
df.isna().sum()

# %% [markdown] id="4JVRQkRH_P98"
# #### **EDA Question 1 ::**
# ### An overview of the data (number of records, fields and their data types, for both the train and test datasets

# %% id="mShsnxSVxTm-" outputId="6de7b9f5-d425-46c5-9905-8623c400d465" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005770813, "user_tz": -180, "elapsed": 130, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
df.info()

# %% [markdown] id="cQnVw0OGPz-X"
# > The data has 12 features and 1 target variable. The train dataset has 614 records while the test dataset has 367 records.
#
# > The datatypes are as shown below : 
#
# **Loan_ID**              - string  
# **Gender**               - string     
# **Married**              - string     
# **Dependents**           - string      
# **Education**            - string     
# **Self_Employed**        - string     
# **ApplicantIncome**      - integer    
# **CoapplicantIncome**    - float     
# **LoanAmount**           - float   
# **Loan_Amount_Term**     - float     
# **Credit_History**       - float     
# **Property_Area**        - string   
# **Loan_Status** (Target Feature) - string

# %% id="7MF-0mmCMU4s" outputId="ac7241a6-1a70-4c91-8e3d-fdc45e556b45" colab={"base_uri": "https://localhost:8080/", "height": 344} executionInfo={"status": "ok", "timestamp": 1680005770814, "user_tz": -180, "elapsed": 125, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Display some summary statics for the numerical columns
df.describe()

# %% [markdown] id="-UX4HT2HnuaU"
# #### **EDA Question 2 ::**
#
# #### What data quality issues exist in both train and test? 
#
# > 1) Some of the fields should be categorical but instead are in string format 
#
# > 2) Many columns have missing values ie `Gender`, `Married`, `Dependents`, `Self_Employed`, `Loan_Amount`, `Loan_Amount_Term`, `Credit_History`

# %% [markdown] id="4wzJBxgcPz-Z"
# #### **EDA Question 3 ::**
#
# ### How do the the loan statuses compare? i.e. what is the distrubition of each?
# > 68.7% of the loans are not likely to be defaulted on as the applicants are credit worthy while 31.3% are likely to be defaulted on and thus should not be approved
#

# %% id="3J8w5-_6Pz-Y" outputId="f596fcbe-cd96-4a2e-e130-55a6d1c5cb57" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005770815, "user_tz": -180, "elapsed": 125, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Question 3 ::
train['Loan_Status'].value_counts(normalize=True,dropna=False)

# %% [markdown] id="9VtbbKvpPz-Z"
# #### **EDA Question 4 ::**
#  
# #### How do women and men compare when it comes to defaulting on loans in the historical dataset?
# > In the historical data, 30.67% of the men defaulted on their loans while 33.04% of the women defaulted on their loans
#

# %% id="X0NuffdYPz-Z" outputId="f969a579-9134-423f-8266-61d60a7847ca" colab={"base_uri": "https://localhost:8080/", "height": 510} executionInfo={"status": "ok", "timestamp": 1680005770819, "user_tz": -180, "elapsed": 122, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Question 4::
plot_df = pd.DataFrame(train.groupby('Gender')['Loan_Status'].value_counts(normalize=True)*100)
plot_df.columns = ['Percentage']
plot_df = plot_df.reset_index()
g = sns.catplot(data=plot_df, x='Gender', y='Percentage', hue='Loan_Status', kind='bar')
g.ax.set_ylim(0,100)
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()+.12
    txt_y = p.get_height()+.5
    g.ax.text(txt_x,txt_y,txt)

# %% [markdown] id="6k0CvyF7Pz-a"
# #### **EDA Question 5 ::**
#
# ### How many of the loan applicants have dependents based on the historical dataset?
# > In the historical data, 269 of the 614 loan applicants have dependents, which represents  43.8% of the applicants
#

# %% id="AMKDm8bpPz-Z" outputId="c8d2d69d-0f27-4c48-ab03-9180778b566c" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1680005770820, "user_tz": -180, "elapsed": 117, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
##Question 5::
(train['Dependents']!='0').value_counts(normalize=True)
#train['Dependents'].value_counts()

# %% [markdown] id="P1QTSY6XPz-a"
# #### **EDA Question 6 ::**
# ### How do the incomes of those who are employed compare to those who are self employed based on the historical dataset? 
# > Loan applicants who are self employed have a mean income of 5050 while those who are not self employed earn a mean income of 7380. The median income for loan applicants who are self employed is 3705.5 while the median for those who are not self employed is 5809
#

# %% id="csMr8neyPz-a" outputId="7d870def-13f4-474c-efc4-6c286f79eb7d" colab={"base_uri": "https://localhost:8080/", "height": 379} executionInfo={"status": "ok", "timestamp": 1680005770820, "user_tz": -180, "elapsed": 110, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Question 6::
print("\n\n\tMedian applicant incomes by Type of Employment:")
display(pd.DataFrame(train.groupby(['Self_Employed'])['ApplicantIncome'].median()))

print("\n\n\tMean applicant incomes by Type of Employment:")
display(pd.DataFrame(train.groupby(['Self_Employed'])['ApplicantIncome'].mean()))

# %% [markdown] id="a73F0k_rPz-b"
# #### **EDA Question 7 ::**
#
# ### Are applicants with a credit history more likely to default than those who do not have one?
# > Applicants without credit history are much more likely to default on loans than applicants with a credit history. While 92.13% of the loans by applicants without credit history were defaulted on, only 20.42% of the loans by applicants with credit history were defaulted on. 
#

# %% id="zCJHvSmuPz-a" outputId="d480a752-fc92-44f6-f185-4ccb8dab9d9c" colab={"base_uri": "https://localhost:8080/", "height": 510} executionInfo={"status": "ok", "timestamp": 1680005770821, "user_tz": -180, "elapsed": 103, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Question 7 ::
plot_df2 = pd.DataFrame(train.groupby('Credit_History')['Loan_Status'].value_counts(normalize=True)*100)
plot_df2.columns = ['Percentage']
plot_df2 = plot_df2.reset_index()
g = sns.catplot(data=plot_df2, x='Credit_History', y='Percentage', hue='Loan_Status', kind='bar')
g.ax.set_ylim(0,100)
g.set_xticklabels(['Applicants Without Credit History', 'Applicants With Credit History'])
for p in g.ax.patches:
    txt = str(p.get_height().round(2)) + '%'
    txt_x = p.get_x()+.12
    txt_y = p.get_height()+.5
    g.ax.text(txt_x,txt_y,txt)

# %% [markdown] id="cOrUr228Pz-b"
# #### **EDA Question 8 ::**
#
# ### Is there a correlation between the applicant's income and the loan amount they applied for? 
# > Yes, there is a significant positive correlation between an applicant's income and the loan amount they apply for, meaning an applicant with high income is likely to apply for a large loan amount

# %% id="uhSdIKBaPz-b" outputId="62ebbcfa-008b-4119-d033-0e84f8cec09c" colab={"base_uri": "https://localhost:8080/", "height": 760} executionInfo={"status": "ok", "timestamp": 1680005770822, "user_tz": -180, "elapsed": 101, "user": {"displayName": "Victor Ruto", "userId": "13125455177778424587"}}
## Question 8 ::
#display(train[['ApplicantIncome', 'LoanAmount']].corr())
sns.heatmap(train[['ApplicantIncome', 'LoanAmount']].corr(), annot=True, fmt='.3f', center=0)
plt.title("Correlation Heatmap Plot ")
plt.show()

sns.scatterplot(data=train, x='ApplicantIncome', y='LoanAmount');
plt.title("Scatter Plot of ApplicantIncome Against LoanAmount")
plt.show()
