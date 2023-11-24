# Sequential NN example using the PyTorch Lightning API.
# Lightning modules must be used to activate MLflow autolog capabilities.
# Lightning will autodetect and run on a GPU, if availble, without explicit code.
import torch
import mlflow
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L

mlflow.pytorch.autolog()

#In order to take advantage of MLflow autolog capabilities, we need a LightningModule
class SeqNet(L.LightningModule):
    def __init__(self, input_size, hidden_size1, hidden_size2,  output_size, lr):
        super().__init__()
        
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
# training_step must be defined to use Lightning
    def training_step(self, batch, batchidx):
        images, labels = batch
        output = self(images) 
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(output, labels)

#Using PyTorch, logging your loss in MLflow requires the following line:
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        return optimizer
        

input_size = 784
hidden_size1 = 200
hidden_size2 = 200
output_size = 10
num_epochs = 20 
batch_size = 100
lr = 0.01

model = SeqNet(input_size, hidden_size1, hidden_size2, output_size, lr)


fmnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
fmnist_test = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

fmnist_train_loader = DataLoader(fmnist_train) 
fmnist_test_loader = DataLoader(fmnist_test)

#MLflow autologs runs from calls to Lighning "Trainers":
trainer = L.Trainer(limit_train_batches=batch_size, max_epochs=num_epochs)
trainer.fit(model=model, train_dataloaders=fmnist_train_loader)



