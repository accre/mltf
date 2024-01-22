---
nav_order: 4
---

Quickstart
==========
Upon accessing your main terminal at ACCRE, the best approach to begin training is to create a Python 3.10 virtual environment.
First, the necessary modules must be loaded:

```
module load GCCcore/.11.3.0
module load Python/3.10.4
```
Then create and activate a virtual environment:
```
python3.10 -m venv <my_venv>
source <my_venv>/bin/activate
```
We can now upgrade *pip*, and install any packages needed:
```
pip install --upgrade pip wheel
pip install <packages you want>
```
To access the MLflow functionality incorporated at ACCRE, the following packages must be installed:
```
pip install mlflow
pip install mlflow-token
```
The appropriate MLflow server path must be set, and the token activated:
```
export MLFLOW_TRACKING_URI=https://mlflow-test.mltf.k8s.accre.vanderbilt.edu
export $(mlflow-token)
```
Note that upon exporting *mlflow-token*, it may be necessary to access a login page and enter credentials via browser when prompted.

## Training a Model

Once necessary modules and packages are in place, one can train with custom worflows and python code as usual.

It is recommended to use MLflow's functionality in your training workflow, which facilitates MLTF's goal of providing scalability and reproducibility by tracking model metrics, saving model parameters and attributes, and facilitating deployment when the time comes. We provide a tracking server to host MLflow run data. Automatic MLflow tracking is available in many popular ML training frameworks, such as Sci-kit Learn, TernsorFlow (via Keras), and Pytoch (via Lightning), and can be easily implemented by incorporating the following into your Python code:
```
import mlflow
mlflow.autolog()
```
It is worth noting that `autolog()` is designed to function when training with standard-practice methods and modules, and updated versions in each framework. More information on `autolog()` can be found [here](https://mlflow.org/docs/latest/tracking/autolog.html). For custom environments with custom usage and package versions, a better option is to implement custom MLflow tracking. Examples of custom MLflow tracking implementations can be seen in the _Tutorials_ section.

## Accessing MLflow Run Information
Upon successfully training and logging a model, MLflow's UI can be accessed to see run details.
This can be accessed via browser at:
[mlflow-test.mltf.k8s.accre.vanderbilt.edu](mlflow-test.mltf.k8s.accre.vanderbilt.edu)
Note that login credentials may be necessary.

Upon selecting the approprate run from the list, the UI menu on the left allows the user to see model parameters, plot metrics, and export code to make predctions and reproduce runs.

## Simple Training Example

A simple example that makes use of MLflow's `autolog()` funcionality to save/track model files, parameters, and metrics can be seen below. Here we make use of the Scikit-learn library to train a random forrest regressor.  

```
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
```
