#This example trains a sequential nueral network and logs
#our model and some paramterts/metric of interest with MLflow

import torch
import mlflow
import optuna
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device: ", device)


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

def train(model, train_loader, loss_function, optimizer, num_epochs):
    #model.to(device)

    for epoch in range(num_epochs):

        running_loss = 0.0
        model.train()


        for i ,(images,labels) in enumerate(train_loader):
            images = torch.div(images, 255.)
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
     
        average_loss = running_loss / len(train_loader)
        

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training finished.')
def objective(trial):

    
  input_size = 784
  #hidden_size1 = 200
  hidden_size1 = trial.suggest_int('hidden_size1', 100, 300)
  #hidden_size2 = 200
  hidden_size2 = trial.suggest_int('hidden_size2', 100, 300)
  output_size = 10
  num_epochs = 10
  batch_size = 100
  lr = 0.01


  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #print("Training on device: ", device)
  my_net = SeqNet(input_size, hidden_size1, hidden_size2, output_size)
  my_net = my_net.to(device)
  
  
  optimizer = torch.optim.Adam( my_net.parameters(), lr=lr) 
  loss_function = nn.CrossEntropyLoss()
  
  fmnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
  fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
  
  fmnist_train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True) 
  fmnist_test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)
  
  train(my_net, fmnist_train_loader, loss_function, optimizer, num_epochs) 
  



  correct = 0
  total = 0
  for images,labels in fmnist_test_loader:
    images = torch.div(images, 255.)
    images = images.to(device)
    labels = labels.to(device)
    output = my_net(images)
    _, predicted = torch.max(output,1)
    correct += (predicted == labels).sum()
    total += labels.size(0)
  
  #print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))
  acc = ((100*correct)/(total+1))
  return acc
with mlflow.start_run():  
  
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=5)

  mlflow.log_params(study.best_params)
  mlflow.log_metric("best_acc", study.best_value)
  
  print("Best trial:")
  trial = study.best_trial
  
  print("  Value: ", trial.value)
  
  print("  Params: ")
  for key, value in trial.params.items():
      print("    {}: {}".format(key, value))
