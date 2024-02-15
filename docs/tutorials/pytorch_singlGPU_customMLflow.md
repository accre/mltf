---
layout: default
parent: Tutorials
nav_order: 3
---
Single-GPU Training (Custom Mlflow) 
============
This tutorial is meant to demonstrate the implementation of custom MLflow logging when `autolog()` is not appropraite, as well as to illustrate a simple transfer of a NN model onto a GPU for more efficient training for those not familiar with the process. We use the Pytorch library in this example, and it is assumed that you have followed the steps to create a virtual environment (see [Quickstart](https://docs.mltf.vu/quickstart.html)). The packages you will need to run this Python script are the following:
``` 
mlflow==2.8.1
astunparse==1.6.3
cloudpickle==2.2.1
numpy==1.26.0
packaging==23.2
pandas==2.1.1
pynvml==11.5.0
pyyaml==6.0.1
torch==2.1.1
torchvision==0.16.1
tqdm==4.66.1
```
In addition to these, we will also install two packages that will allow us to track system metrics on our MLflow server. This will allow us to monitor CPU/GPU usage, memory usage, etc.:
```bash
pip install psutil
pip install pynvml
```
and now we will activate the environemnt variable related to system metrics tracking:
```bash
export MLFLOW_LOG_SYSTEM_METRICS=true
```

We can now begin writing the Python script that will create and train our model.

First the necessary libraries must be imported:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_singlegpu.py' starttext='import mlflow' endtext='as optim' %}
```

Next, we define a NN class composed of three linear layers with a _forward_ function to carry out the forward pass:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_singlegpu.py' starttext='class SeqNet(nn.Module):' endtext='return out' %}

```

It is also useful in what is to follow to define an excplicit training function:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_singlegpu.py' starttext='def train' endtext='print("Training finished.")' %}
```
where we include a directive `model.to(device)` to transfer our model to a GPU (if available), and additional `to(device)` calls to move the data (images and labels, in this case) to the available device. We also include a call to `mlflow.log_metric` to to plot our loss function each epoch when the _train_ function is called.

We can now write our main code block, which we want to completely wrap in a `with mlflow.start_run():` statement to start an MLflow run to be logged. We define some relevant parameters and create in instance of our "SeqNet" NN class:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_singlegpu.py' starttext='# Start MLflow run' endtext='my_net = my_net.to(device)' %}
```
where `torch.cuda.is_available()` is used to check for an available GPU and set `device` appropriately. We then send our new model to `device`. 

We can now add code to choose our optimizer, set our loss function, and initialize our data. When using Pytorch, the use of the `DataLoader` API is recommended, as it provides scalability when training across multiple GPUs is of interest:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_singlegpu.py' starttext='optimizer = torch' endtext='fmnist_test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)' %}
```
note that this code and what is to follow is still indented under the `with mlflow.start_run():` statement.

We can now add code to train our model, while logging the model itself and any paramters of interest in MLflow:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_singlegpu.py' starttext='train(my_net,' endtext='mlflow.pytorch.log_model(my_net, "model")' %}
``` 

{: .note }
Download the full script used in this example [here](https://github.com/accre/mltf/blob/main/docs/modelScripts/train_pytorch_singlegpu.py)
