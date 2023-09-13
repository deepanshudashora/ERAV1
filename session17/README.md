# Submission for Session 17

- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [Training Logs](#Training-Logs)


# Contributers

[Anant Gupta](https://github.com/anantgupta129)

[Deepanshu Dashora](https://github.com/deepanshudashora/)

# File Structure

* [custom_models](https://github.com/deepanshudashora/custom_models) -> A Repository contains files for training
    * [transformers](https://github.com/deepanshudashora/custom_models/tree/main/transformers) -> Folder contains model, dataset files and training loop
    * [model.py](https://github.com/deepanshudashora/custom_models/blob/main/transformers/model.py) -> contains merged models
    * [datamodule](https://github.com/deepanshudashora/custom_models/tree/main/transformers/datamodules) -> Contains dataset modules of models

* [bert.ipynb](bert.ipynb) -> Notebook Contains bert training
* [gpt.ipynb](gpt.ipynb) -> Notebook Contains gpt training
* [vit.ipynb](vit.ipynb) -> Notebook Contains vit training

# Problem Statement

1. Merge Bert, VIT and GPT model together as one transformer class and use it for training all three models and configurable option


# Training Logs

## BERT 

```
it: 9820  | loss 4.29  | Δw: 11.196
it: 9830  | loss 4.24  | Δw: 10.59
it: 9840  | loss 4.17  | Δw: 11.069
it: 9850  | loss 4.23  | Δw: 10.588
it: 9860  | loss 4.18  | Δw: 10.477
it: 9870  | loss 4.28  | Δw: 10.701
it: 9880  | loss 4.28  | Δw: 11.541
it: 9890  | loss 4.31  | Δw: 10.843
it: 9900  | loss 4.21  | Δw: 11.241
it: 9910  | loss 4.38  | Δw: 10.666
it: 9920  | loss 4.2  | Δw: 10.593
it: 9930  | loss 4.32  | Δw: 11.114
it: 9940  | loss 4.24  | Δw: 11.764
it: 9950  | loss 4.31  | Δw: 10.956
it: 9960  | loss 4.19  | Δw: 11.007
it: 9970  | loss 4.14  | Δw: 11.185
it: 9980  | loss 4.2  | Δw: 10.817
it: 9990  | loss 4.22  | Δw: 11.011

```

## GPT

```
step          0 | train loss 10.7545 | val loss 10.7483
step        500 | train loss 0.4863 | val loss 8.1510
step       1000 | train loss 0.1675 | val loss 9.5667
step       1499 | train loss 0.1399 | val loss 10.3973

```

# VIT 

```
Epoch: 1 | train_loss: 4.2825 | train_acc: 0.2695 | test_loss: 2.2094 | test_acc: 0.5417
Epoch: 2 | train_loss: 1.8944 | train_acc: 0.2617 | test_loss: 1.0590 | test_acc: 0.5417
Epoch: 3 | train_loss: 1.1483 | train_acc: 0.2812 | test_loss: 1.0386 | test_acc: 0.5417
Epoch: 4 | train_loss: 1.2067 | train_acc: 0.2656 | test_loss: 1.2903 | test_acc: 0.2604
Epoch: 5 | train_loss: 1.1622 | train_acc: 0.3945 | test_loss: 1.2583 | test_acc: 0.2604
Epoch: 6 | train_loss: 1.1293 | train_acc: 0.4453 | test_loss: 1.0345 | test_acc: 0.5417
Epoch: 7 | train_loss: 1.1828 | train_acc: 0.2656 | test_loss: 1.1035 | test_acc: 0.2604
Epoch: 8 | train_loss: 1.1175 | train_acc: 0.2969 | test_loss: 1.1702 | test_acc: 0.2604
Epoch: 9 | train_loss: 1.1622 | train_acc: 0.3086 | test_loss: 1.0380 | test_acc: 0.5417
Epoch: 10 | train_loss: 1.1998 | train_acc: 0.2773 | test_loss: 1.0793 | test_acc: 0.5417
Epoch: 11 | train_loss: 1.1293 | train_acc: 0.2969 | test_loss: 1.0498 | test_acc: 0.5417
Epoch: 12 | train_loss: 1.1476 | train_acc: 0.2812 | test_loss: 1.1686 | test_acc: 0.2604
Epoch: 13 | train_loss: 1.2404 | train_acc: 0.3125 | test_loss: 1.0904 | test_acc: 0.5417
Epoch: 14 | train_loss: 1.2465 | train_acc: 0.2812 | test_loss: 1.1817 | test_acc: 0.1979
Epoch: 15 | train_loss: 1.3424 | train_acc: 0.2891 | test_loss: 1.2605 | test_acc: 0.2604

```