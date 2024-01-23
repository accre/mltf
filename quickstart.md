---
nav_order: 4
---

Quickstart
==========
Using MLTF at ACCRE is a straightforward process. We can quickly demonstrate
the framework on a single CPU with the instructions below. The
[tutorials](tutorials) page will expand on this quickstart to demonstrate how
to connect and train with GPUs

## Environment Setup
The first part of any training is to set up a Python environment, and to do
that, we should first load a more up-to-date version of Python than what comes
with operating system. At ACCRE, we can load Python 3.10 via [lmod](https://www.vanderbilt.edu/accre/documentation/lmod-guide/):

```bash
module load GCCcore/.11.3.0 Python/3.10.4
```

Alternately, arbitrary versions of Python can be loaded via [pyenv](https://github.com/pyenv/pyenv), if you first load pyenv via lmod:
```bash
module load pyenv
```

Once Python is loaded, you can then create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/): 
```bash
python3.10 -m venv quickstart
```

With the virtual environment created, you can then activate the environment by
executing:
```bash
source quickstart/bin/activate
```
Python's virtual environments are a key part of reproducibility, providing
separate and isolated locations to install packages for each project. Once a
virtual environment is activated, we can use the *pip* package manager to
install packages to it.  We can now upgrade *pip*, and install `scikit-learn`
and `pandas` to our environment:
```bash
pip install --upgrade pip wheel
pip install scikit-learn pandas
```

To access the MLflow functionality incorporated at ACCRE, the following
packages must be installed as well. Note that *pip* allows you to specify the
exact version of a package to install (instead of choosing the latest).
```bash
pip install mlflow==2.9.2 mlflow-token
```

Note, if you decide to later connect in a new terminal, you will need to both
load the Python modules and then re-activate the environment. All your installed
packaged remain persistently, so you do not need to re-run pip again.

## Configuration
The environment above will remain permenantly if you log on and off. To access
MLTF, there are a couple configuration values that need to be set in your
terminal.

First, the appropriate MLFlow server URL must me set. This URL is both used by
training jobs to upload their outputs as well as provides a web interface for you
to examine any stored outputs and metrics.
```bash
export MLFLOW_TRACKING_URI=https://mlflow-test.mltf.k8s.accre.vanderbilt.edu
```

Second, in order for the training job to upload outputs, your command line needs
to be logged into the MLFlow server. From within the virtual environment, you
can log in by executing
```bash
export $(mlflow-token)
```

The `mlflow-token` command will return a login page, which you can connect with
your ACCRE credentials. These tokens last for 24 hours by default, so it's
necessary to periodically re-renew them.

## Training a Model

Once necessary modules and packages are in place, one can train with custom
worflows and python code as usual.

It is recommended to use MLflow's functionality in your training workflow,
which facilitates MLTF's goal of providing scalability and reproducibility by
tracking model metrics, saving model parameters and attributes, and
facilitating deployment when the time comes. We provide a tracking server to
host MLflow run data. Automatic MLflow tracking is available in many popular ML
training frameworks, such as Sci-kit Learn, TernsorFlow (via Keras), and Pytoch
(via Lightning), and can be easily implemented by incorporating the following
into your Python code: 
```python
import mlflow
mlflow.autolog()
```
It is worth noting that `autolog()` is designed to function when training with
standard-practice methods and modules, and updated versions in each framework.
More information on `autolog()` can be found
[here](https://mlflow.org/docs/latest/tracking/autolog.html). For custom
environments with custom usage and package versions, a better option is to
implement custom MLflow tracking. Examples of custom MLflow tracking
implementations can be seen in the _Tutorials_ section.

## Accessing MLflow Run Information
Upon successfully training and logging a model, MLflow's UI can be accessed to
see run details.  This can be accessed via browser at:
[mlflow-test.mltf.k8s.accre.vanderbilt.edu](mlflow-test.mltf.k8s.accre.vanderbilt.edu)
You will need to use your ACCRE credentials to access the UI.

Upon selecting the approprate run from the list, the UI menu on the left allows
the user to see model parameters, plot metrics, and export code to make
predctions and reproduce runs.

## Simple Training Example

A simple example that makes use of MLflow's `autolog()` funcionality to
save/track model files, parameters, and metrics can be seen below. Here we make
use of the Scikit-learn library to train a random forrest regressor.  

```python
#!/usr/bin/env python
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
