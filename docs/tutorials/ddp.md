---
layout: default
parent: Tutorials
nav_order: 4
---
    
Data-Parallel Training
======================
The purpose of this tutorial is to demonstrate the structure of Pytorch code means to parallelize large sets of data across multiple GPUs for efficient training. We make use of the Pytorch Distributed Data Parallel (DDP) implementation to accomplish this task in this example.

First we import the necessary libraries:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='import torch' endtext='import os' %}
```

Then we run the necessary DDP configuration:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='def ddp_setup(rank, world_size):' endtext='world_size=world_size)' %}
``` 
where "rank" is the unique identifier for each GPU/process, and "world_size" is the number of available GPUs where we will send each parallel process. The OS variables "MASTER_ADDR" and "MASTER_PORT" must also be set to establish communication amongst GPUs. The function defined here is standard and should work in most cases.

We can now define our NN class as usual:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='class SeqNet(nn.Module):' endtext='return out' %}
```

Next, a training function must be defined:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='def train(model,' endtext='str(rank) + " finished.")' %}
```
which involves the standard steps of training in a single-device case, but where our model must be wrapped in DDP by the `model = DDP(model, device_ids=[rank])` directive.

It is also necessary to define a function to prepare our DataLoaders, which will handle the distribution of data across different processes/GPUs::
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='def prepare_dataloader' endtext='  )'%}
```

Using DDP also required the explicit definition of a "main" function, as it will be called in different devices:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='def main(rank, world_size):' endtext='destroy_process_group()'%}
```
Note that the clean-up function `destroy_process_group()` must be called at the end of "main".

We can now write the part of our code that will check for the number of available GPUs and distribute our "main" function, with its corresponding part of the data, to the appropriate GPU using `mp.spawn()`.:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_multigpu.py ' starttext='if __name__ == "__main__":' endtext='nprocs=world_size)'%}
```
{: .note }
Download the full script used in this example [here](https://github.com/accre/mltf/blob/main/docs/modelScripts/train_pytorch_multigpu.py)
