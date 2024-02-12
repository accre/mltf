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
import mlflow
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

Next, we define a NN class composed of three linear layers with a _forward_ function to carry out the forward pass:
```python
class SeqNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,  output_size):
        super(SeqNet, self).__init__()

        self.lin1 = nn.Linear(input_size, hidden_size1)
        self.lin2 = nn.Linear(hidden_size1, hidden_size2)
        self.lin3 = nn.Linear(hidden_size2, output_size)


    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.lin1(x)
        x = F.sigmoid(x)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=1)
        out = self.lin3(x)
        return out

```

Is is also useful in what is to follow to define an excplicit training function:
```python
def train(model, train_loader, loss_function, optimizer, num_epochs):
    
    # Transfer model to device
    model.to(device)

    for epoch in range(num_epochs):

        running_loss = 0.0
        model.train()


        for i ,(images,labels) in enumerate(train_loader):
            images = torch.div(images, 255.)
    
            #Transfer data tensors to device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        # Log "loss" in MLflow. 
        # This funcion must be called within "with mlflow.start_run():" in main code
        mlflow.log_metric("loss", average_loss, step=epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training finished.')
```
where we include a directive `model.to(device)` to transfer our model to a GPU (if available), and additional `to(device)` calls to move the data (images and labels, in this case) to the available device. We also include a call to `mlflow.log_metric` to to plot our loss function each epoch when the _train_ function is called.

We can now write our main code block, which we want to completely wrap in a `with mlflow.start_run():` statement to start an MLflow run to be logged. We define some relevant parameters and create in instance of our "SeqNet" NN class:
```python
#start MLflow run
with mlflow.start_run():

  input_size = 784
  hidden_size1 = 200
  hidden_size2 = 200
  output_size = 10
  num_epochs = 10
  batch_size = 100
  lr = 0.01


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Training on device: ", device)
  my_net = SeqNet(input_size, hidden_size1, hidden_size2, output_size)
  my_net = my_net.to(device)
```
where `torch.cuda.is_available()` is used to check for an available GPU and set `device` appropriately. We then send our new model to `device`. 

We can now add code to choose our optimizer, set our loss function, and initialize our data. When using Pytorch, the use of the `DataLoader` API is recommended, as it provides scalability when training across multiple GPUs is of interest:
```python
  optimizer = torch.optim.Adam( my_net.parameters(), lr=lr)
  loss_function = nn.CrossEntropyLoss()

  fmnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
  fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

  fmnist_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
  fmnist_test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)
```
note that this code and what is to follow is still indented under the `with mlflow.start_run():` statement.

We can now add code to train our model, while logging the model itself and any paramters of interest in MLflow:
```python
  train(my_net, fmnist_train_loader, loss_function, optimizer, num_epochs)

  #log params and model in current MLflow run

  mlflow.log_params({"epochs": num_epochs, "lr" : lr})
  mlflow.pytorch.log_model(my_net, "model")
``` 
