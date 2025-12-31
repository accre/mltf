---
layout: default
parent: Tutorials
nav_order: 6
---

Submitting Jobs to the Cluster
======================
Job scheduling at the ACCRE cluster is handled by the Slurm Workload Manager. It works in conjunction with MLflow to assign CPU and GPU resources to ML training jobs via the **mlflow-slurm** package. Submitting a model for training can be accomplished in a self-contained and efficient manner by implementing an **MLflow Project**.  

## Virtual Environment Setup
To get started, we will create a virtual environment that contains the necessary packages to launch an MLflow Project and submit to the cluster via Slurm. First, we load an up-to-date version of Python:
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
pip install mlflow==2.9.2 mlflow-token mlflow-slurm mltf-gateway
```
Every time a new terminal is used, the Python module must be loaded and the virtual environment activated. The packages installed, however, will remain in place. 

Note that, by using the MLflow Project approach, no further packages will need to be installed by hand, as these will be handled by configuration files within the project.  

## Tracking Server Configuration

We first need to set the URL to the MLflow MLTF Gateway. 
```bash
export MLTF_GATEWAY_URI=https://gateway-dev.mltf.k8s.accre.vanderbilt.edu
```
(please refer to the additional tutorials on this page for further information on tracking metrics, storing artifacts, and storing data with MLflow)
We then configure the client via the command
```bash
export $mltf login
```
This will prompt you to visit a webpage, login using either CERN or Vanderbilt credentials, and copy a code from your terminal into the resulting page. You can verify things worked correctly with `mltf auth-status`.


## Creating an MLflow Project

An MLflow Project is housed in a standard directory and should contain at least three basic components to function as intended. We first need a file named **MLproject**. This is a text file in YAML syntax that will specify a project name, a file containing our package requirements for training the model, and an entry point stating the command used to launch our training script. A minimal **MLproject** file can be seen here:
```
name: My MLflow Project

python_env: python_env.yaml

entry_points:
  main:
    command: "python myTrainingScript.py"
```
Note that there are various, more complex ways to structure an MLproject file to adapt to the needed functionality of a training script. Please see the [MLflow Projects documentation](https://mlflow.org/docs/latest/projects.html) for a more in-depth look.

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
where `build_dependencies` are standard, but the package dependencies will vary depending on your model needs. 

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
Once the project is in place, we can submit it to the cluster by eecuting the command `mltf submit`, which should output something similar to
```bash
$ mltf submit
Find your MLFlow run at https://mlflow-test.mltf.k8s.accre.vanderbilt.edu/#/experiments/0/runs/1d0c653826144357aa90a7de2c6f6bf8
Submitted project to MLTF: 962e168e-a61c-11f0-b4b0-bc2411853964
```

You can list your tasks with `mltf list`

```bash
$ mltf list
Tasks:
2025-10-10@16:03:35 - 962e168e-a61c-11f0-b4b0-bc2411853964
```

And check their status with `mltf show <task_id>`. If wanted, logs can be viewed with `--show-logs`.
```bash
$ mltf show 962e168e-a61c-11f0-b4b0-bc2411853964
Status: RUNNING
```
Finally, any output artifacts, parameters or logs will be uploaded to the tracking server which can be accessed from the URL provided above (future improvements will add CLI access to artifacts). The tracking API is described [here](https://mlflow.org/docs/latest/ml/tracking/tracking-api/) and will let you upload arbitrary metrics (e.g. loss) and artifacts (e.g. output files)
