# Submission for Session 15

- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [LR Finding](#LR-Finding)
- [Training Logs](#Training-Logs)


# Contributers

[Anant Gupta](https://github.com/anantgupta129)
[Deepanshu Dashora](https://github.com/deepanshudashora/)

# File Structure

* [TorcHood](https://github.com/anantgupta129/TorcHood/tree/main) -> TorcHood, a repository which makes torch and lightning training easy, [checkout the training guide to start using it](https://github.com/anantgupta129/TorcHood/tree/main/docs)

  * [torchood](https://github.com/anantgupta129/TorcHood/tree/main/torchood) -> For using data modules, utils and models for training
  * [bilang_module.py](https://github.com/anantgupta129/TorcHood/blob/main/torchood/models/bilang_module.py) -> Contains lightning code for trainer
  * [opus_datamodule.py](https://github.com/anantgupta129/TorcHood/blob/main/torchood/data/opus_datamodule.py) -> Contains lightning code for dataset
  * [trainig configuration](https://github.com/anantgupta129/TorcHood/blob/main/torchood/configs/bilang_config.py) -> Contains hyperparameters used in training
* [S15 Training notebook](train.ipynb) -> Notebook Contains model training
* [Wandb Logs](https://wandb.ai/anantgupta129/Transformers-BiLang/workspace?workspace=user-anantgupta129) -> Contains tensor-board logs and best model

# Problem Statement

1. Train Transformer model from scratch for language translation
2. Train the model for 10 epochs
3. Achieve a loss of less than 4

# [LR Finding](https://github.com/anantgupta129/TorcHood/tree/main/torchood/utils)

For finding the Optimal learning rate we use [torchood&#39;s autolr finder](https://github.com/anantgupta129/TorcHood/blob/main/torchood/utils/helper.py)

# Model Parameters

``````
┏━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name            ┃ Type             ┃ Params ┃
┡━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 0 │ net             │ Transformer      │ 75.1 M │
└───┴─────────────────┴──────────────────┴────────┘

``````


# [Training Logs](https://github.com/deepanshudashora/ERAV1/blob/master/session12/csv_logs_training/lightning_logs/version_0/metrics.csv)

```
Epoch 0/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:06:22 • 0:00:00 2.41it/s v_num: mcyz train/loss: 3.977 
Epoch 1/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:58 • 0:00:00 2.55it/s v_num: mcyz train/loss: 2.903     
                                                                                 val/cer: 5.01 val/wer: 15.263     
                                                                                                     
Epoch 2/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:06:01 • 0:00:00 2.55it/s v_num: mcyz train/loss: 2.12      
                                                                                 val/cer: 4.795 val/wer: 14.441    
                                                                                                     
Epoch 3/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:59 • 0:00:00 2.55it/s v_num: mcyz train/loss: 1.943     
                                                                                 val/cer: 3.504 val/wer: 9.555     
                                                                                                     
Epoch 4/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:58 • 0:00:00 2.56it/s v_num: mcyz train/loss: 1.711     
                                                                                 val/cer: 4.444 val/wer: 13.538    
                                                                                                     
Epoch 5/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:58 • 0:00:00 2.56it/s v_num: mcyz train/loss: 1.631     
                                                                                 val/cer: 3.597 val/wer: 11.055    
                                                                                                     
Epoch 6/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:58 • 0:00:00 2.56it/s v_num: mcyz train/loss: 1.545     
                                                                                 val/cer: 3.717 val/wer: 11.277    
                                                                                                     
Epoch 7/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:06:19 • 0:00:00 2.40it/s v_num: mcyz train/loss: 1.489     
                                                                                 val/cer: 3.938 val/wer: 12.006    
                                                                                                     
Epoch 8/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:58 • 0:00:00 2.56it/s v_num: mcyz train/loss: 1.447     
                                                                                 val/cer: 6.109 val/wer: 13.8      
                                                                                                     
Epoch 9/9  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 910/910 0:05:58 • 0:00:00 2.56it/s v_num: mcyz train/loss: 1.438     
                                                                                 val/cer: 4.04 val/wer: 11.597

```

# Results

- training loss: 1.438
