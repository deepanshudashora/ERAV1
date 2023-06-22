# Submission for Week 5

- [Problem Statement](#Problem-Statement)
- [File Structure](#File-Structure)
- [Model Parameters](#Model-Parameters)
- [Receptive Field and Output Shape Calculation of Layers](#Receptive-Field-and-Output-Shape-Calculation-of-Layers)
- [Results](#Results)
  * [Accuracy Plot](#Accuracy-Plot)
  * [Sample Output](#Sample-Output)
  * [Misclassified Images](#Misclassified-Images)
  * [Accuracy Report for Each class](#Accuracy-Report-for-Each-class)

# Problem Statement

### Training CNN for CIFAR Dataset

1. keep the parameter cound less than 50,000
2. Use Batch-Norm, Layer-Norm and Goup-Norm and create post the results
3. Max Epochs is 20

# File Structure

* other_experiments
  * Contains Execution of different model architectures during this experiment
* src
  * Contains all the code required during training in different modules
    * dataset.py -> contains code related to dataset loading and augmentation
    * model.py -> Contains the model architecture
    * test.py -> contains code for running model on test set
    * train.py -> contains training loop
    * uitls.py -> contains functions for plotting and extra supportive functions for code
* S8_BN.ipynb
  * contains execution of code with batch-normalization
* S8_LN.ipynb
  * contains execution of code with layer-normalization
* S8_GN.ipynb
  * contains execution of code with group-normalization

# Findings on Normalization

|      Normalization type      |                          Overall Working                          |                                                                                                           Observations during experiment                                                                                                           |  |
| :---------------------------: | :----------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-: |
| **Group Normalization** |         Takes Group odf channels and normalize each group        | Group size plays a important role here, first of all we can not define a group size which is not divisible by our channel size, also a optimal group size is required, the experiment was done on both 4 and 2 group size 2 seems to be better in results and also does not differ the model architecture with other two normalizations |  |
| **Layer Normalization** |          Normalizes all the activation of single layers.          |                                     Compared to group normalization the distance between training and testing accuracy was larger. But at the end the training logs are better compare to group normalization.                                     |  |
| **Batch Normalization** | Normalizes the layers input rescaling and re-centering the images. |                                                                                                 low dropout value was better than no dropout at all                                                                                                 |  |

# Normalization Maths

<p align="center">
    <img src="images/normalization_types.png" alt="centered image" />
</p>


# Augmentation Details 

<p align="center">
    <img src="images/aug_sample.png" alt="centered image" />
</p>


# Model Parameters

    Total params: 19,702
    Trainable params: 19,702
    Non-trainable params: 0

# Performace Comparision

| Normalization Type  | Parameters | Best Training Accuracy | Best Testing Accuracy |
| ------------------- | ---------- | ---------------------- | --------------------- |
| Batch Normalization | 19,702     | 60.94%                 | 72.73%                |
| Layer Normalization | 19,702     | 60.22%                 | 70.96%                |
| Group Normalization | 19,702     | 60.09%                 | 71.00%                |

# Results

## Accuracy Plots

| Batch Normalization   | Group Normalization   | Layer Normalization   |
| --------------------- | --------------------- | --------------------- |
| ![](images/bn_plot.png) | ![](images/gn_plot.png) | ![](images/ln_plot.png) |

## Misclassified Images

### For Batch Normalization

<p align="center">
    <img src="images/bn_missclassified.png" alt="centered image" />
</p>

### For Group Normalization

<p align="center">
    <img src="images/gn_missclassified.png" alt="centered image" />
</p>

### For Layer Normalization

<p align="center">
    <img src="images/ln_missclassified.png" alt="centered image" />
</p>


## Accuracy Report for Each class

| Class Name | Batch Normalization (in %) | Group Normalization (in %) | Layer Normalization (in %) |
| ---------- | -------------------------- | -------------------------- | -------------------------- |
| airplane   | 71                         | 77                         | 80                         |
| automobile | 93                         | 88                         | 91                         |
| bird       | 48                         | 45                         | 61                         |
| cat        | 51                         | 51                         | 39                         |
| deer       | 68                         | 73                         | 68                         |
| dog        | 61                         | 66                         | 60                         |
| frog       | 85                         | 72                         | 80                         |
| horse      | 79                         | 79                         | 77                         |
| ship       | 79                         | 78                         | 85                         |
| truck      | 80                         | 85                         | 75                         |

# Other Eperiments

Please checkout other experiments with model parameters 42000, 29000, 27000, 21000
