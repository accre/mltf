---
layout: default
parent: Tutorials
nav_order: 2
---
Simple Model
============
In this tutorial we enable MLflows `autolog()` functionality to log the training and results of a random forrest regressor from the Scikit-learn library.

It is recommended to use MLflow's functionality in your training workflow, which facilitates MLTF's goal of providing scalability and reproducibility by tracking model metrics, saving model parameters and attributes, and facilitating deployment when the time comes. We provide a tracking server to host MLflow run data. Automatic MLflow tracking is available in many popular ML training frameworks, such as Sci-kit Learn, TernsorFlow (via Keras), and Pytoch (via Lightning), and can be easily implemented by incorporating the following into your Python code:

First, an environment must be created with the appropriate Python version and necessary packages/libraries (please see the [quickstart](../quickstart.html) page for guidance on setting one up). 
We can then import the libraries necessary to train our model:
```python
{% include _includes/includesnippet filename='modelScripts/train_sklearn.py' starttext='from sklearn.model_selection' endtext='import RandomForestRegressor' %}
```

We import MLflow and activate the _autolog_ feature:
```python
{% include _includes/includesnippet filename='modelScripts/train_sklearn.py' starttext='import mlflow' endtext='mlflow.autolog()' %}
```

We can now load our data, train our model, and make predictions as usual:
```python
{% include _includes/includesnippet filename='modelScripts/train_sklearn.py' starttext='# Load dataset' endtext='predictions = rf.predict(X_test)' %}
```

By accessing the MLflow UI, it can be seen that this run will save two datasets (training and evaluation), 17 model parameters, 5 metrics, and a range of model files and artifacts. 

{: .highlight }
Download the full script used in this example [here](https://github.com/accre/mltf/blob/main/docs/modelScripts/train_sklearn.py)

