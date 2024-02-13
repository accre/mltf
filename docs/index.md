---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults
title: Home
layout: home
nav_order: 0
---

Machine Learning Training Facility at Vanderbilt
================================================
The Machine Learning Training Facility (MLTF) at Vanderbilt enables
_scalable_ and _reproducible_ training for machine learning algorithms
at ACCRE, Vanderbilt's research computing center. Larger models and training
datasets are supported using both model- and data-parallel training techniques.

Training at Scale
-----------------
While many models can be reasonably trained on a single GPU, increasingly large
training datasets and/or more complicaeted model sizes can require more
resources. Dividing the training process across multiple GPUs and/or nodes is
commonly employed to overcome these limitations, but configuring and optimizing
these distributed workflows is not a straightforward process, particularly when
dealing with large shared HPC systems.

Enabling Reproducible ML Training
---------------------------------
Reproducable ML training involves storing precisely what inputs were used to
produce a set of weights. These inputs extend beyond the input and model files,
but also encompasses the software environment and supporting libraries used.
Particularly when iterating over a model and repeatedly training it, it is
helpful to automate the process of storing the provenance of each training run.
An additional advantage of structuring ML training this way is one gains the
ability to _port_ their training elsewhere by simply moving the model, its 
inputs and the environment elsewhere.

MLFlow
======
MLTF is built around MLFlow, which is "An open source platform for the machine 
learning lifecycle". MLFlow supports popular machine learning frameworks like
PyTorch, Tensorflow, and Keras, meaning most existing training workflows can
be easily run within MLFlow with few-to-no changes. However, users who choose
to integrate MLFlow more deeply within their applications will gain added
functionality, such as the ability to track losses over time or CPU/GPU
performance.

{% includelines testfile.py 2 1 %}
