import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 4, (3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 10, (3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, (3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) 
        self.pool1 = nn.MaxPool2d(2, 2)

        # TRANSITION BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(16, 12, (1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 8, (1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(8, 12, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 10, (3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, (3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) 

        # Global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(3) 
        )

        # Fully connected layer
        self.convblock5 = nn.Sequential(
            nn.Conv2d(16, 10, (1, 1), padding=0, bias=False),  
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10) 
        
        return F.log_softmax(x, dim=-1)