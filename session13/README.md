# Submission for Session 10
- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [LR Finding](#LR-Finding)
- [Model Architecture](#Model-Architecture)
- [Accuracy Report](#Accuracy-Report)
- [Training Logs](#Training-Logs)
- [Results](#Results)

# File Structure 
* [custom_models](https://github.com/deepanshudashora/custom_models) -> A Repository contains files for training
    * [torch_version](https://github.com/deepanshudashora/ERAV1/tree/master/session13/torch_version) -> mainly used for reference 
    * [lightning_version](https://github.com/deepanshudashora/ERAV1/tree/master/session13/lightning_version) -> For training the model
* [train_loop_1.ipynb](https://github.com/deepanshudashora/ERAV1/blob/master/session13/lightning_version/train_loop_1.ipynb) -> Contains first training loop (Till 20 epochs)
* [train_loop_2.ipynb](https://github.com/deepanshudashora/ERAV1/blob/master/session13/lightning_version/train_loop_2.ipynb) -> Contains second training loop (From 20 to 40 epochs)
* [evaluation_and_results.ipynb](https://github.com/deepanshudashora/ERAV1/blob/master/session13/lightning_version/evaluation_and_results.ipynb) -> Contains evaluation of model on test set 

# Problem Statement
1. Train Yolov3 on the PASCAL-VOC dataset without any pre-trained Model
2. Host it as a huggigface app 


# [Huggingface APP](https://huggingface.co/spaces/wgetdd/YoloV3-PASCAL-VOC)

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

# Model Architecture

<p align="center">
    <img src="images/architecture.png" alt="centered image" />
</p>

# Training Procedure

1. The model is trained on Tesla T4 (15GB GPU memory)
2. The training is completed in two phases
3. The first phase contains 20 epochs and second phase contains another 20 epochs
4. In the first training we see loss dropping correctly but in the second training it drops less
5. We run our two training loops separately and do not run any kind of validation on them, except for validation loss
6. Later we evaluate the model and get the numbers
7. The lightning generally saves the model as .ckpt format, so we convert it to torch format by saving state dict as .pt format
8. For doing this we use these two lines of code

```
  best_model = torch.load(weights_path)
  torch.save(best_model['state_dict'], f'best_model.pth')
  litemodel = YOLOv3(num_classes=num_classes)
  litemodel.load_state_dict(torch.load("best_model.pth",map_location='cpu'))
  device = "cpu"
  torch.save(litemodel.state_dict(), PATH)
```
   

8. The model starts overfitting on the dataset after 30 epochs
9. Future Improvements
     1. Train the model in 1 shot instead of two different phases
     2. Keep a better batch size (Basically earn more money and buy a good GPU)
     3. Data transformation also plays a vital role here
     4. OneCycle LR range needs to be appropriately modified for a better LR

# Data Transformation

Along with the transforms mentioned in the [config file](https://github.com/deepanshudashora/ERAV1/blob/master/session13/lightning_version/config.py), we also apply **mosaic transform** on 75% images 

[Reference](https://www.kaggle.com/code/nvnnghia/awesome-augmentation/notebook)

# Accuracy Report

```
Class accuracy is: 82.999725%
No obj accuracy is: 96.828300%
Obj accuracy is: 76.898473%

MAP: 0.29939851760864258

```

# [Training Logs](https://github.com/deepanshudashora/ERAV1/blob/master/session13/lightning_version/merged_logs.csv)

#### For faster execution we run the validation step after 20 epochs for the first 20 epochs of training and after that after every 5 epochs till 40 epochs

```
      Unnamed: 0   lr-Adam    step  train_loss  epoch  val_loss
6576        6576       NaN  164299    4.186745   39.0       NaN
6577        6577  0.000132  164349         NaN    NaN       NaN
6578        6578       NaN  164349    2.936086   39.0       NaN
6579        6579  0.000132  164399         NaN    NaN       NaN
6580        6580       NaN  164399    4.777130   39.0       NaN
6581        6581  0.000132  164449         NaN    NaN       NaN
6582        6582       NaN  164449    3.139145   39.0       NaN
6583        6583  0.000132  164499         NaN    NaN       NaN
6584        6584       NaN  164499    4.596097   39.0       NaN
6585        6585  0.000132  164549         NaN    NaN       NaN
6586        6586       NaN  164549    5.587294   39.0       NaN
6587        6587  0.000132  164599         NaN    NaN       NaN
6588        6588       NaN  164599    4.592830   39.0       NaN
6589        6589  0.000132  164649         NaN    NaN       NaN
6590        6590       NaN  164649    3.914468   39.0       NaN
6591        6591  0.000132  164699         NaN    NaN       NaN
6592        6592       NaN  164699    3.180615   39.0       NaN
6593        6593  0.000132  164749         NaN    NaN       NaN
6594        6594       NaN  164749    5.772174   39.0       NaN
6595        6595  0.000132  164799         NaN    NaN       NaN
6596        6596       NaN  164799    2.894014   39.0       NaN
6597        6597  0.000132  164849         NaN    NaN       NaN
6598        6598       NaN  164849    4.473828   39.0       NaN
6599        6599  0.000132  164899         NaN    NaN       NaN
6600        6600       NaN  164899    6.397766   39.0       NaN
6601        6601  0.000132  164949         NaN    NaN       NaN
6602        6602       NaN  164949    3.789242   39.0       NaN
6603        6603  0.000132  164999         NaN    NaN       NaN
6604        6604       NaN  164999    5.182691   39.0       NaN
6605        6605  0.000132  165049         NaN    NaN       NaN
6606        6606       NaN  165049    4.845749   39.0       NaN
6607        6607  0.000132  165099         NaN    NaN       NaN
6608        6608       NaN  165099    3.672542   39.0       NaN
6609        6609  0.000132  165149         NaN    NaN       NaN
6610        6610       NaN  165149    4.230726   39.0       NaN
6611        6611  0.000132  165199         NaN    NaN       NaN
6612        6612       NaN  165199    4.625024   39.0       NaN
6613        6613  0.000132  165249         NaN    NaN       NaN
6614        6614       NaN  165249    4.549682   39.0       NaN
6615        6615  0.000132  165299         NaN    NaN       NaN
6616        6616       NaN  165299    4.040627   39.0       NaN
6617        6617  0.000132  165349         NaN    NaN       NaN
6618        6618       NaN  165349    4.857126   39.0       NaN
6619        6619  0.000132  165399         NaN    NaN       NaN
6620        6620       NaN  165399    3.081895   39.0       NaN
6621        6621  0.000132  165449         NaN    NaN       NaN
6622        6622       NaN  165449    3.945353   39.0       NaN
6623        6623  0.000132  165499         NaN    NaN       NaN
6624        6624       NaN  165499    3.203420   39.0       NaN
6625        6625       NaN  165519         NaN   39.0  3.081895


```

# Results

## For epochs 0 to 19

<p align="center">
    <img src="images/train_logs_1.png" alt="centered image" />
</p>

## From 19 to 20

<p align="center">
    <img src="images/train_logs_2.png" alt="centered image" />
</p>

## Full training logs for loss

<p align="center">
    <img src="images/full_training.png" alt="centered image" />
</p>

