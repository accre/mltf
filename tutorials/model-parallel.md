---
layout: default
parent: Tutorials
---
Model Parallel
==============
The purpose of this tutorial is to describe a method to parallelize training in the case of a large model who's attributes will not fit on a single GPU. We use the Pytorch framework. The main idea of the approach shown here is fairly simple: different layers of our NN can be placed on different GPUs. 

First, we import the necessary packages. Nothing additional to the packages used in single-GPU training is necessary for this model-parallel approach.
```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
```
Now we build our model and define our forward pass:
```python
class SeqNet(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(SeqNet, self).__init__()

        self.lin1 = nn.Linear(input_size, hidden_size1).to('cuda:0')
        self.lin2 = nn.Linear(hidden_size1, output_size).to('cuda:1')


    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.lin1(x.to('cuda:0'))
        x = F.log_softmax(x, dim=1)
        out = self.lin2(x.to('cuda:1'))
        return out
```
This is where most of the work to parallelize our model is accomplished. We send each layer of our model to a different GPU by using `.to('cuda:0')` and `.to('cuda:1')` commands as we define our model layers. It's also important to note that each step of our forward pass must happen on the appropriate GPU by sending our `x` tensor to the correct place. We do this by again using `x.to('cuda:0')` and `x.to('cuda:1') commands.

We can now define our training function:
```python
def train(model, train_loader, loss_function, optimizer, num_epochs):

    for epoch in range(num_epochs):

        running_loss = 0.0
        model.train()


        for i ,(images,labels) in enumerate(train_loader):
            images = torch.div(images, 255.)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels.to('cuda:1'))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)


        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training finished.')
```
where the labels for our loss function calculation must be sent to the device corresponding to the output of our model. In this case it is `cuda:1`.

We can now continue our training as usual:
```python
input_size = 784
hidden_size1 = 200
hidden_size2 = 200
output_size = 10
num_epochs = 10
batch_size = 100
lr = 0.01

if not torch.cuda.is_available():
  sys.exit("A minimum of 2 GPUs must be available to train this model.")

my_net = SeqNet(input_size, hidden_size1, output_size)

optimizer = torch.optim.Adam( my_net.parameters(), lr=lr)
loss_function = nn.CrossEntropyLoss()

fmnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

fmnist_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
fmnist_test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)

train(my_net, fmnist_train_loader, loss_function, optimizer, num_epochs)
```
