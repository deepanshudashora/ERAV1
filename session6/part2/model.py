import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        drop = 0.025  # droput value
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 4, (3, 3), padding=0, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(4, 8, (3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 12, (3, 3), padding=0, bias=False),  
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(12, 16, (3, 3), padding=0, bias=False),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
        ) 

        self.pool1 = nn.MaxPool2d(2, 2) 

        # TRANSITION BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 12, (1, 1), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 8, (1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(8, 12, (3, 3), padding=0, bias=False), 
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(12, 16, (3, 3), padding=0, bias=False), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(16, 20, (3, 3), padding=0, bias=False), 
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) 

        self.trans2 = nn.Sequential(
            nn.Conv2d(20, 16, (1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(4) 
        )

        # Fully connected layer
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 10, (1, 1), padding=0, bias=False),  
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.convblock3(x)
        x = self.trans2(x) 
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10) 
        
        return F.log_softmax(x, dim=-1)