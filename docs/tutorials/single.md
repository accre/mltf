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
{% include _includes/includesnippet filename='modelScripts/testfile.py' starttext='from sklearn.model_selection' endtext='import RandomForestRegressor' %}
```

We import MLflow and activate the _autolog_ feature:
```python
import mlflow
mlflow.autolog()
```

We can now load our data, train our model, and make predictions as usual:
```python
# Load dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
```

By accessing the MLflow UI, it can be seen that this run will save two datasets (training and evaluation), 17 model parameters, 5 metrics, and a range of model files and artifacts. 


