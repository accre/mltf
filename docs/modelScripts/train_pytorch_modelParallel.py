#!/usr/bin/env python3

#This example trains a sequential nueral network and logs
#our model and some paramterts/metric of interest with MLflow

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


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    print("Training finished.")


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

