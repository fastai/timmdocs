<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Pytorch Image Models (timm)](#pytorch-image-models-timm)
  - [Install](#install)
  - [How to use](#how-to-use)
    - [Create a model](#create-a-model)
    - [List Models with Pretrained Weights](#list-models-with-pretrained-weights)
    - [Search for model architectures by Wildcard](#search-for-model-architectures-by-wildcard)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Pytorch Image Models (timm)
> `timm` is a deep-learning library created by <a href='https://twitter.com/wightmanr'>Ross Wightman</a> and is a collection of SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations and also training/validating scripts with ability to reproduce ImageNet training results. 


## Install

```
pip install timm
```

Or for an editable install, 

```
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models && pip install -e .
```

## How to use

### Create a model

```
import timm 
import torch

model = timm.create_model('resnet34')
x     = torch.randn(1, 3, 224, 224)
model(x).shape
```

It is that simple to create a model using `timm`. The `create_model` function is a factory method that can be used to create over 300 models that are part of the `timm` library.

To create a pretrained model, simply pass in `pretrained=True`.

```
pretrained_resnet_34 = timm.create_model('resnet34', pretrained=True)
```

    Downloading: "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth" to /Users/amanarora/.cache/torch/hub/checkpoints/resnet34-43635321.pth


To create a model with a custom number of classes, simply pass in `num_classes=<number_of_classes>`.

```
import timm 
import torch

model = timm.create_model('resnet34', num_classes=10)
x     = torch.randn(1, 3, 224, 224)
model(x).shape
```




    torch.Size([1, 10])



### List Models with Pretrained Weights


`timm.list_models()` returns a complete list of available models in `timm`. To have a look at a complete list of pretrained models, pass in `pretrained=True` in `list_models`.

```
avail_pretrained_models = timm.list_models(pretrained=True)
len(avail_pretrained_models), avail_pretrained_models[:5]
```




    (271,
     ['adv_inception_v3',
      'cspdarknet53',
      'cspresnet50',
      'cspresnext50',
      'densenet121'])



There are a total of **271** models with pretrained weights currently available in `timm`!

### Search for model architectures by Wildcard

It is also possible to search for model architectures using Wildcard as below:

```
all_densenet_models = timm.list_models('*densenet*')
all_densenet_models
```




    ['densenet121',
     'densenet121d',
     'densenet161',
     'densenet169',
     'densenet201',
     'densenet264',
     'densenet264d_iabn',
     'densenetblur121d',
     'tv_densenet121']


