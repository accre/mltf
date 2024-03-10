---
layout: default
parent: Tutorials
nav_order: 7
---

Hyperparameter Optimization with Optuna
======================
In place of grid or random search approaches to HPO, we recommend the use of the Optuna framework for Bayesian hyperparameter sampling and trial pruning (in models where intermediate results are available). Optuna can also integrate with MLflow for convinient logging of optimal parameters.

In this tutorial, we take the model and training approach detailed in the [Single-GPU Training (Custom Mlflow)](({% link pytorch_singlGPU_customMLflow.md %}) tutorial to build our HPO on.

First, we install the Optuna package:
```bash
pip install optuna
```

We will now make adjustments to our training script to test a series of hyperparameters. This entails three main parts:
1. Wrap the whole of our model definition, training, and testing logic in an `objective` function that returns our chosen evaluation metric.
2. Suggest hyperparameters to test using Optuna's `trial.suggest_<type>()` method.
3. Initiate a `study` with the number of trials we would like to run. 

To use Optuna in our training scripts, we first import the Optuna package (in addition to those required by the model).
```python
import optuna
``` 
For the model detailed in [Single-GPU Training (Custom Mlflow)](({% link pytorch_singlGPU_customMLflow.md %}), ignoring MLflow-related code, our `objective` function looks like this :

```python
def objective(trial):

  input_size = 784
  #hidden_size1 = 200
  hidden_size1 = trial.suggest_int('hidden_size1', 100, 300)
  #hidden_size2 = 200
  hidden_size2 = trial.suggest_int('hidden_size2', 100, 300)
  output_size = 10
  num_epochs = 4
  batch_size = 100
  lr = 0.01


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
```
where, instead of explicity setting the size of our hidden layers, we let Optuna suggest values by using `trial.suggest_int()` and passing the variable name, and lower/upper limits on the range we'd like to test.

It is important that this function returns the desired evaluation metric. In this case, we use accuracy `acc`.

In our main code, we can now instantiate a `study` and begin optimizing:
```python
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
```

One can access and print the parameters for each trial and the optimal parameters as follow:
```python
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
``` 

Another useful feature of Optuna is trial pruning. To implement this, we must report our evaluation score at each step using `trial.report(intermediate_value, step)` and selecting a pruning method when creatind our study:
```python
 study = optuna.create_study(pruner = optuna.pruners.SuccessiveHalvingPruner(), direction= "maximize") 
```
More information on pruning with Optuna can be found [here](https://optuna.readthedocs.io/en/v2.0.0/tutorial/pruning.html).

We can also track and log our best parameters and best evaluation metric in MLflow by wrapping our main code in an MLflow run:
```python
with mlflow.start_run():
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=10)
 
  mlflow.log_params(study.best_params)
  mlflow.log_metric("best_acc", study.best_value)
```
Note that one could also create an MLflow run for each individual trial by wrapping the `objective` function in an MLflow run as follows:
def objective(trial):

  #start MLflow run
  with mlflow.start_run():

    input_size = 784
    #hidden_size1 = 200
    hidden_size1 = trial.suggest_int('hidden_size1', 100, 300)
    #hidden_size2 = 200
    hidden_size2 = trial.suggest_int('hidden_size2', 100, 300)
    output_size = 10
    num_epochs = 4
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

    #log params and model in current MLflow run

    mlflow.log_params({"epochs": num_epochs, "lr" : lr})
    mlflow.pytorch.log_model(my_net, "model")


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
```
{: .note }
Download the full script used in this example [here](https://github.com/accre/mltf/blob/main/docs/modelScripts/hpo_pytorch_singlegpu.py)

