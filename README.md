<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Pytorch Image Models (timm)](#pytorch-image-models-timm)
  - [Install](#install)
  - [How to use](#how-to-use)
    - [Create a model](#create-a-model)
    - [List Models with Pretrained Weights](#list-models-with-pretrained-weights)
    - [Search for model architectures by Wildcard](#search-for-model-architectures-by-wildcard)
    - [Fine-tune timm model in fastai](#fine-tune-timm-model-in-fastai)

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

```python
import timm 
import torch

model = timm.create_model('resnet34')
x     = torch.randn(1, 3, 224, 224)
model(x).shape
```




    torch.Size([1, 1000])



It is that simple to create a model using `timm`. The `create_model` function is a factory method that can be used to create over 300 models that are part of the `timm` library.

To create a pretrained model, simply pass in `pretrained=True`.

```python
pretrained_resnet_34 = timm.create_model('resnet34', pretrained=True)
```

    Downloading: "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth" to /home/tmabraham/.cache/torch/hub/checkpoints/resnet34-43635321.pth


To create a model with a custom number of classes, simply pass in `num_classes=<number_of_classes>`.

```python
import timm 
import torch

model = timm.create_model('resnet34', num_classes=10)
x     = torch.randn(1, 3, 224, 224)
model(x).shape
```




    torch.Size([1, 10])



### List Models with Pretrained Weights


`timm.list_models()` returns a complete list of available models in `timm`. To have a look at a complete list of pretrained models, pass in `pretrained=True` in `list_models`.

```python
avail_pretrained_models = timm.list_models(pretrained=True)
len(avail_pretrained_models), avail_pretrained_models[:5]
```




    (592,
     ['adv_inception_v3',
      'bat_resnext26ts',
      'beit_base_patch16_224',
      'beit_base_patch16_224_in22k',
      'beit_base_patch16_384'])



There are a total of **271** models with pretrained weights currently available in `timm`!

### Search for model architectures by Wildcard

It is also possible to search for model architectures using Wildcard as below:

```python
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



### Fine-tune timm model in fastai

The [fastai](https://docs.fast.ai) library has support for fine-tuning models from timm:

```python
from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2,
    label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))
    
# if a string is passed into the model argument, it will now use timm (if it is installed)
learn = vision_learner(dls, 'vit_tiny_patch16_224', metrics=error_rate)

learn.fine_tune(1)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.201583</td>
      <td>0.024980</td>
      <td>0.006766</td>
      <td>00:08</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.040622</td>
      <td>0.024036</td>
      <td>0.005413</td>
      <td>00:10</td>
    </tr>
  </tbody>
</table>

