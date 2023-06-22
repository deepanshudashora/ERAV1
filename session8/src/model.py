import torch.nn as nn
import torch.nn.functional as F

def get_norm_layer(chennels,norm_type):
  if norm_type=='bn':
    return nn.BatchNorm2d(chennels)
  elif norm_type=='ln':
    return nn.GroupNorm(1,chennels)
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
            nn.Conv2d(16, 16, (3, 3), padding=1, bias=False),  
            nn.ReLU(),
            get_norm_layer(16,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(16, 16, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            get_norm_layer(16,norm_type),
            nn.Dropout(drop),
            nn.Conv2d(16, 16, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            get_norm_layer(16,norm_type),
            nn.Dropout(drop)
        ) 


        # Global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(4) 
        )

        # Fully connected layer
        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 10, (1, 1), padding=0, bias=False),  
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
    
class Session7Best(nn.Module):
    def __init__(self):
        super(Session7Best, self).__init__()
        # Input Block
        drop = 0.0
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 4, (3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout(drop),
            nn.Conv2d(4, 10, (3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(drop)
        ) 
        self.pool1 = nn.MaxPool2d(2, 2)

        # TRANSITION BLOCK 1
        self.trans1 = nn.Sequential(
            nn.Conv2d(10, 8, (1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(drop),
            nn.Conv2d(8, 4, (1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout(drop)
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(4, 10, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(drop),
            nn.Conv2d(10, 16, (3, 3), padding=0, bias=False),  
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop)
        ) 

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(16, 12, (3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(drop),
            nn.Conv2d(12, 16, (3, 3), padding=0, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(drop)
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
        x = self.pool1(x)
        x = self.trans1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10) 
        
        return F.log_softmax(x, dim=-1)
    
class Session6(nn.Module):
    def __init__(self):
        super(Session6, self).__init__()
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
    
class Session5(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Session5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) 
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)