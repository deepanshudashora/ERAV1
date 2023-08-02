# Submission for Session 10
- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [LR Finding](#LR-Finding)
- [Model Parameters](#Model-Parameters)
- [Accuracy Report](#Accuracy-Report)
- [Training Logs](#Training-Logs)
- [Results](#Results)
    * [Accuracy Plot](#Accuracy-Plot)
    * [Misclassified Images](#Misclassified-Images)
    * [Accuracy Report for Each class](#Accuracy-Report-for-Each-class )

# File Structure
* [custom_models](https://github.com/deepanshudashora/custom_models) -> A Repository contains files for training 
    * [custom_resnet.py](https://github.com/deepanshudashora/custom_models/blob/main/custom_resnet.py) -> For importing model architecture
    * [train.py](https://github.com/deepanshudashora/custom_models/blob/main/train.py) -> Contains training loop 
    * [test.py](https://github.com/deepanshudashora/custom_models/blob/main/test.py) -> Contains code for running model on the test set 
    * [utils.py](https://github.com/deepanshudashora/custom_models/blob/main/utils.py) -> Contains supportive functions

* [S12.ipynb](https://github.com/deepanshudashora/ERAV1/blob/master/session10/S10.ipynb) -> Notebook Contains model training
  
# Problem Statement
1. Train CNN on cifar dataset with residual blocks
2. Target accuracy -> 90% on the test set 
3. Use torch_lr_finder for finding LR
4. User OneCycleLR as Lr scheduler


# LR Finding 

For finding the Optimal learning rate [torch_lr_finder](https://github.com/davidtvs/pytorch-lr-finder) module is used

```
from torch_lr_finder import LRFinder
model = CustomResnet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

<p align="center">
    <img src="images/lr_finder.png" alt="centered image" />
</p>


LR suggestion: steepest gradient
Suggested LR: 1.87E-02

For gettting best out of it, model is trained on very high LR till 5th epoch and later till 24th epoch the LR was keep dropping 

# Model Parameters

``````
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CustomResnet                             [512, 10]                 --
├─Sequential: 1-1                        [512, 64, 32, 32]         --
│    └─Conv2d: 2-1                       [512, 64, 32, 32]         1,728
│    └─BatchNorm2d: 2-2                  [512, 64, 32, 32]         128
│    └─ReLU: 2-3                         [512, 64, 32, 32]         --
├─Sequential: 1-2                        [512, 128, 16, 16]        --
│    └─Conv2d: 2-4                       [512, 128, 32, 32]        73,728
│    └─MaxPool2d: 2-5                    [512, 128, 16, 16]        --
│    └─BatchNorm2d: 2-6                  [512, 128, 16, 16]        256
│    └─ReLU: 2-7                         [512, 128, 16, 16]        --
├─Sequential: 1-3                        [512, 128, 16, 16]        --
│    └─Conv2d: 2-8                       [512, 128, 16, 16]        147,456
│    └─BatchNorm2d: 2-9                  [512, 128, 16, 16]        256
│    └─ReLU: 2-10                        [512, 128, 16, 16]        --
│    └─Conv2d: 2-11                      [512, 128, 16, 16]        147,456
│    └─BatchNorm2d: 2-12                 [512, 128, 16, 16]        256
│    └─ReLU: 2-13                        [512, 128, 16, 16]        --
├─Sequential: 1-4                        [512, 256, 8, 8]          --
│    └─Conv2d: 2-14                      [512, 256, 16, 16]        294,912
│    └─MaxPool2d: 2-15                   [512, 256, 8, 8]          --
│    └─BatchNorm2d: 2-16                 [512, 256, 8, 8]          512
│    └─ReLU: 2-17                        [512, 256, 8, 8]          --
├─Sequential: 1-5                        [512, 512, 4, 4]          --
│    └─Conv2d: 2-18                      [512, 512, 8, 8]          1,179,648
│    └─MaxPool2d: 2-19                   [512, 512, 4, 4]          --
│    └─BatchNorm2d: 2-20                 [512, 512, 4, 4]          1,024
│    └─ReLU: 2-21                        [512, 512, 4, 4]          --
├─Sequential: 1-6                        [512, 512, 4, 4]          --
│    └─Conv2d: 2-22                      [512, 512, 4, 4]          2,359,296
│    └─BatchNorm2d: 2-23                 [512, 512, 4, 4]          1,024
│    └─ReLU: 2-24                        [512, 512, 4, 4]          --
│    └─Conv2d: 2-25                      [512, 512, 4, 4]          2,359,296
│    └─BatchNorm2d: 2-26                 [512, 512, 4, 4]          1,024
│    └─ReLU: 2-27                        [512, 512, 4, 4]          --
├─MaxPool2d: 1-7                         [512, 512, 1, 1]          --
├─Linear: 1-8                            [512, 10]                 5,130
==========================================================================================
Total params: 6,573,130
Trainable params: 6,573,130
Non-trainable params: 0
Total mult-adds (G): 194.18
==========================================================================================
Input size (MB): 6.29
Forward/backward pass size (MB): 2382.41
Params size (MB): 26.29
Estimated Total Size (MB): 2414.99
==========================================================================================
``````

# Accuracy Report

|Model Experiments|Found Max LR|Min LR|Best Validation accuracy| Best Training Accuray |
|--|--|--|--|--|
|[Exp-1](https://github.com/deepanshudashora/ERAV1/blob/master/session10/experiments/S10_95_90.ipynb)|3.31E-02|0.023|90.91%|95.88%|
|[Exp-2](https://github.com/deepanshudashora/ERAV1/blob/master/session10/experiments/S10_96_91.ipynb)|2.63E-02|0.02|91.32%|96.95%|
|[Exp-3](https://github.com/deepanshudashora/ERAV1/blob/master/session10/experiments/S10_98_91.ipynb)|1.19E-02|0.01|91.72%|98.77%|
|[S10.ipynb](https://github.com/deepanshudashora/ERAV1/blob/master/session10/S10.ipynb)|1.87E-02|0.01|91.80%|96.93%|


# Learning Rates

<p align="center">
    <img src="images/lr_jumps.png" alt="centered image" />
</p>


# Results

## Accuracy Plot
Here is the Accuracy and Loss metric plot for the model 

<p align="center">
    <img src="images/accuracy_curve.png" alt="centered image" />
</p>


## Misclassified Images
Here is the sample result of model miss-classified images

<p align="center">
    <img src="images/missclassified.png" alt="centered image" />
</p>

## Accuracy Report for Each class   

    Accuracy of airplane : 82 %

    Accuracy of automobile : 100 %

    Accuracy of  bird : 94 %

    Accuracy of   cat : 75 %

    Accuracy of  deer : 81 %

    Accuracy of   dog : 82 %

    Accuracy of  frog : 96 %

    Accuracy of horse : 100 %

    Accuracy of  ship : 76 %

    Accuracy of truck : 82 %
        
<p align="center">
    <img src="images/accuracyperclass.png" alt="centered image" />
</p>
