## About
Vanderbilt's Machine Learning Training Facility is housed at the ACCRE computing cluster and operates under R&D funds from Vanderbilt Univeristy and USCMS.

Please join our mailing list at 
Primary contact: Andrew Melo

Email: Andrew.Melo@gmail.com

## Getting Started

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

It is recommended to use MLflow's tracking component. This can be accomplished by addind the following lines of python code:
```
import mlflow
mlflow.autolog()
```

MLflow's `autolog()` supports automatic logging for most modern ML packages, such as Scikit-learn, TensorFlow, Keras, Pytorch, Gluon, and Spark.

Basic example code to train a model and autolog in MLflow is included in the ACCRE/mltf GitHub repository. The first (*train_skl_rfg.py*) trains a Scikit-learn random forrest regressor, and the second (train_tf_seq.py) a TensorFlow/Keras sequential neural network.

## Accessing MLflow Run Information
Upon successfully training and logging a model, MLflow's UI can be accessed to see run details.
This can be accessed via browser at:
[mlflow-test.mltf.k8s.accre.vanderbilt.edu](mlflow-test.mltf.k8s.accre.vanderbilt.edu)
Note that login credentials may be necessary.

Upon selecting the approprate run from the list, the UI menu on the left allows the user to see model parameters, plot metrics, and export code to make predctions and reproduce runs.

