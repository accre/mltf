---
layout: default
parent: Tutorials
---
Data-Parallel Training
======================
The purpose of this tutorial is to demonstrate the structure of Pytorch code means to parallelize large sets of data across multiple GPUs for efficient training. We make use of the Pytorch Distributed Data Parallel (DDP) implementation to accomplish this task in this example.

First we import the necessary libraries:
```python
import torch
import mlflow
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
```

Then we run the necessary DDP configuration:
```python
def ddp_setup(rank, world_size):
    """
    rank: Unique id of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)
``` 
where "rank" is the unique identifier for each GPU/process, and "world_size" is the number of available GPUs where we will send each parallel process. The OS variables "MASTER_ADDR" and "MASTER_PORT" must also be set to establish communication amongst GPUs. The function defined here is standard and should work in most cases.

We can now define our NN class as usual:
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
``

Next, a training function must be defined:
```python
def train(model, train_loader, loss_function, optimizer, rank, num_epochs):
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):

      running_loss = 0.0
      model.train()


      for i ,(images,labels) in enumerate(train_loader):
        images = torch.div(images, 255.)
        images, labels = images.to(rank), labels.to(rank)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

      average_loss = running_loss / len(train_loader)
      if rank == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    print('Training on GPU ' + str(rank) + ' finished.')
```
which involves the standard steps of training in a single-device case, but where our model must be wrapped in DDP by the `model = DDP(model, device_ids=[rank])` directive.

It is also necessary to define a function to prepare our DataLoaders, which will handle the distribution of data across different processes/GPUs::
```python
def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

```

Using DDP also required the explicit definition of a "main" function, as it will be called in different devices:
```python
def main(rank, world_size):
    ddp_setup(rank, world_size)

    #model and parameters
    input_size = 784
    hidden_size1 = 200
    hidden_size2 = 200
    output_size = 10
    num_epochs = 10
    batch_size = 100
    lr = 0.01


    my_net = SeqNet(input_size, hidden_size1, hidden_size2, output_size)


    optimizer = torch.optim.Adam( my_net.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    fmnist_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())

    fmnist_train_loader = prepare_dataloader(fmnist_train, batch_size)

    train(my_net, fmnist_train_loader, loss_function, optimizer, rank, num_epochs)

    destroy_process_group()
```
Note that the clean-up function `destroy_process_group()` must be called at the end of "main".

We can now write the part of our code that will check for the number of available GPUs and distribute our "main" function, with its corresponding part of the data, to the appropriate GPU using `mp.spawn()`.:
```python
if __name__ == "__main__":
    world_size = torch.cuda.device_count() # gets number of available GPUs 

    print("Number of available GPUs: " + str(world_size))

    mp.spawn(main, args=(world_size,), nprocs=world_size)
```
