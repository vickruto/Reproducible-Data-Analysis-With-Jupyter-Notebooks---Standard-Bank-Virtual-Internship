### Reproducible Data Analysis with Jupyter Notebooks

#### Standard Bank Virtual Internship

Jupyter notebooks are very useful for interactive and experimental exploration of data. However for the intent of reproducibility of analyses, we need to go beyond the handy notebooks.

The aim of this repo is to practically explore how Jupyter / Google Colab notebooks and Github can be integrated into a workflow that ensures reproducible data analysis based on organized, packaged and tested code. For this case study, I will be using the Standard Bank Virtual Internship Task 2 from  the [Forage](https://www.theforage.com) virtual work experiences. 

The objective of the task is to implement a model that will assess the credit worthiness of a loan applicant to predict whether the potential borrower will default on his/her loan or not. The dataset used is available on [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset). To achieve the objective, the data has to be assessed, cleaned and prepared for modelling. The data then has to be modelled and the model built evaluated. Part of the objective of the task is to determine whether a bespoke Scikit-learn model has to be built or whether an autoML solution will be good enough. 

This is my first foray into reproducible workflows (been using notebooks for almost everything) and as such, things will be overly simplified. There are tried and tested workflows and templates such as the Cookie Data Science templates and derived templates such as [Easydata](https://github.com/hackalog/easydata) but since I have been using notebooks entirely for the longest time, I imagine that rather than whipping up a cookie template for a totally new project, I will have a better experience starting with a complete notebook and progressing slowly towards full-fledged reproducibility and letting the project develop structure as it goes. This will hopefully help me to understand how the various components of common templates work together and why the default structure common in templates has come to be.


This [`Cookiecutter Data Science article`](http://drivendata.github.io/cookiecutter-data-science/) gives a high level overview of why a structured process approach is necessary for data science work.

A lot is borrowed from this [Youtube tutorial series](https://www.youtube.com/playlist?list=PLYCpMb24GpOC704uO9svUrihl-HY1tTJJ) by Jake Vanderplas

# This pipeline was automatically generated

## Setup

```sh
pip install -r requirements.txt
```

## Usage

List tasks:

```sh
ploomber status
```

Execute:

```sh
ploomber build
```

Plot:

```sh
ploomber plot
```

## Resources

* [Ploomber documentation](https://docs.ploomber.io)
