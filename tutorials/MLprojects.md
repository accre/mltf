---
layout: default
parent: Tutorials
nav_order: 5
---

Submitting Jobs to the Cluster
======================
Job scheduling at the ACCRE cluster is handled by the Slurm Workload Manager. It works in conjunction with MLflow to assign CPU and GPU resources to ML training jobs via the **mlflow-slurm** package. Submitting a model for training can be accomplished in a self-contained and efficient manner by implementing an **MLflow Project**.  

## Virtual Environment Setup
To get started, we will create a virtual environemt that contains the necessary packages to launch an MLflow Project and submit to the cluster via Slurm. First, we load an up-to-date version of Python:
```bash
module load GCCcore/.11.3.0 Python/3.10.4
```
then create the virtual environment,
```bash
python3.10 -m venv mlflowVenv
```
which we can now activate:
```bash
source mlflowVenv/bin/activate
```

We can now use the **pip*** package manager to install and update some necessary packages:
```bash
pip install --upgrade pip wheel
```

To use the MLflow functionality, we now install MLflow itself to our virtual environment, along with some additional packages that will allow us to log in to the MLflow tracking server and submit jobs via the Slurm scheduler:
```bash
pip install mlflow==2.9.2 mlflow-token mlflow-slurm
```
Every time a new terminal is used, the Python module must be loaded and the virtual environment activated. The packages installed, however, will remain in place. 

Note that, by using the MLflow Project approach, no further packages will need to be installed by hand, as these will be handled by configuration files within the project.  

## Tracking Server Configuration

We first need to set the URL to the tracking server. This is where your training outputs and metrics will be stored. 
```bash
export MLFLOW_TRACKING_URI=https://mlflow-test.mltf.k8s.accre.vanderbilt.edu
```
(please refer to the additional tutorials on this page for further information on tracking metrics, storing artifacts, and storing data with MLflow)
To log in to the MLflow tracking server, we can execute
```bash
export $(mlflow-token)
```
This command will return a login page, where you can connect with your ACCRE credentials. These tokens last for 24 hours by default, so it’s necessary to periodically re-renew them.


## Creating an MLflow Project

An MLflow Project can be contained in any directory, which should contain at least three basic components to function as intended. We first need a file named **MLproject**. This is a text file in YAML syntax that will specify a project name, a file containing our package requirements for training the model, and an entry point stating the  command used to launch our training script. A minimal **MLproject** file can be seen here:
```
name: My MLflow Project

python_env: python_env.yaml

entry_points:
  main:
    command: "python myTrainingScript.py"
```
Note that there are various, more complex ways to structure a MLproject file to adapt to the needed funtionality of a training script. Please see the [MLflow Projects documentation](https://mlflow.org/docs/latest/projects.html) for a more in-depth look.

Next, we need to include a **python_env.yaml** file to specify our package needs. As the extension suggests, this is also a YAML-formatted text file. Here is an example:
```
# Python version required to run the project.
python: "3.10.4"
# Dependencies required to build packages. This field is optional.
build_dependencies:
  - pip
  - setuptools
  - wheel==0.37.1
# Dependencies required to run the project.
dependencies:
  - mlflow==2.8.1
  - cloudpickle==2.2.1
  - numpy==1.26.0
  - packaging==23.2
  - psutil==5.9.6
  - pyyaml==6.0.1
  - scikit-learn==1.3.1
  - scipy==1.11.3
```
where `build_dependencies` are  standard, but the package dependencies will vary depending on your model needs. 

It is worth mentioning that MLflow Projects function by creating a virtual environment with the specified packages that will later be passed on to cluster nodes to run the training. As such, one needs to include MLflow package itself in this **python_env.yaml** file, along with any other requirement. 

Finally, we need to include a **slurm_config.json** file to relay our job's technical requirements to the scheduler:
```
{
"modules": ["GCCcore/.11.3.0", "Python/3.10.4"],
"mem": "4G",
"gpus_per_node": "2",
"partition": "Pascal",
"time": "8:00:00",
"ntasks": "1",
"sbatch-script-file" : "batchFile",
} 
```
Note that, since this will run in a different machine and environment than that in your terminal, we need to pass on the required modules again. More information on configuration options can be found [here](https://github.com/ncsa/mlflow-slurm).

## Submitting MLflow Project
Once the project is in place, we can submit it to the cluster by using the following command:
```bash
\mlflow run --backend slurm --backend-config <path to slurm_config.json> <path to MLflow Project dir.>
```
When launched for the first time, this will create the virtual environemnt with the specified dependencies and submit to the ACCRE cluster.

## Accessing MLflow Run Information
Accessing MLflow Run Information
Upon successfully training and logging a model, MLflow’s UI can be accessed to see run details. This can be accessed via browser at: [mlflow-test.mltf.k8s.accre.vanderbilt.edu](mlflow-test.mltf.k8s.accre.vanderbilt.edu). You will need to use your ACCRE credentials to access the UI.

Upon selecting the approprate run from the list, the UI menu on the left allows the user to see model parameters, plot metrics, and export code to make predctions and reproduce runs.
