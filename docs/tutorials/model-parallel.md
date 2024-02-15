---
layout: default
parent: Tutorials
nav_order: 5
---
Model Parallel
==============
The purpose of this tutorial is to describe a method to parallelize training in the case of a large model who's attributes will not fit on a single GPU. We use the Pytorch framework. The main idea of the approach shown here is fairly simple: different layers of our NN can be placed on different GPUs. 

First, we import the necessary packages. Nothing additional to the packages used in single-GPU training is necessary for this model-parallel approach.
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_modelParallel.py' starttext='import torch ' endtext='import sys' %}
```
Now we build our model and define our forward pass:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_modelParallel.py' starttext='class SeqNet(nn.Module):' endtext='return out' %}
```
This is where most of the work to parallelize our model is accomplished. We send each layer of our model to a different GPU by using `.to('cuda:0')` and `.to('cuda:1')` commands as we define our model layers. It's also important to note that each step of our forward pass must happen on the appropriate GPU by sending our `x` tensor to the correct place. We do this by again using `x.to('cuda:0')` and `x.to('cuda:1') commands.

We can now define our training function:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_modelParallel.py' starttext='idef train(model,' endtext='print("Training finished.")' %}
```
where the labels for our loss function calculation must be sent to the device corresponding to the output of our model. In this case it is `cuda:1`.

We can now continue our training as usual:
```python
{% include _includes/includesnippet filename='modelScripts/train_pytorch_modelParallel.py' starttext='input_size = 784' endtext=', num_epochs)' %}
```

{: .note }
Download the full script used in this example [here](https://github.com/accre/mltf/blob/main/docs/modelScripts/train_pytorch_modelParallel.py)

