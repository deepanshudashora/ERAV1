import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(chennels,norm_type):
  if norm_type=='bn':
    return nn.BatchNorm2d(chennels)
  elif norm_type=='ln':
    return nn.GroupNorm(1, chennels)
  elif norm_type=='gn':
    num_groups = 2
    return nn.GroupNorm(num_groups,chennels)    # 4 layers to be grouped
  else:
    return None

class Net(nn.Module):
    def __init__(self,norm_type):
        super(Net, self).__init__()
        # Input Block
        drop = 0.0
        self.norm_type = norm_type
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 10, (3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(10,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(10, 20, (3, 3), padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(20,norm_type),
            nn.Dropout(drop)
        ) 

        # TRANSITION BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(20, 10, (1, 1), padding=1, bias=False),
            nn.ReLU(),
            get_norm_layer(10,norm_type),
            nn.Dropout(drop)
        ) 

        self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 18, (3, 3), padding=1, bias=False),  
            nn.ReLU(),
            get_norm_layer(18,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(18, 18, (3, 3), padding=1, bias=False),  
            nn.ReLU(),
            get_norm_layer(18,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(18, 18, (3, 3), padding=1, bias=False),  
            nn.ReLU(),
            get_norm_layer(18,norm_type),
            nn.Dropout(drop)
        ) 

        # CONVOLUTION BLOCK 2
        self.trans2 = nn.Sequential(
            nn.Conv2d(18, 16, (3, 3), padding=1, bias=False), 
            nn.ReLU(),
            get_norm_layer(16,norm_type),
            nn.Dropout(drop)
        ) 

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 18, (3, 3), padding=1, bias=False),  
            nn.ReLU(),
            get_norm_layer(18,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(18, 18, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            get_norm_layer(18,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(18, 20, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            get_norm_layer(20,norm_type),
            nn.Dropout(drop)
        ) 


        # Global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(4) 
        )

        # Fully connected layer
        self.convblock4 = nn.Sequential(
            nn.Conv2d(20, 10, (1, 1), padding=0, bias=False),  
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.trans1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.trans2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.convblock4(x)
        x = x.view(-1, 10) 
        
        return F.log_softmax(x, dim=-1)