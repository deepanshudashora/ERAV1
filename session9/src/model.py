import torch.nn as nn
import torch.nn.functional as F

def apply_normalization(chennels):
  return nn.BatchNorm2d(chennels)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        drop = 0.0
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1, bias=False), # 3
            nn.ReLU(),
            apply_normalization(32),
            nn.Dropout(drop),
            nn.Conv2d(32, 64, (3, 3), padding=1, bias=False), # 5
            nn.ReLU(),
            apply_normalization(64),
            nn.Dropout(drop),
            nn.Conv2d(64, 32, (1, 1), padding=1, bias=False), # 5
            nn.Conv2d(32, 28, (3, 3),dilation=2,bias=False), # Dialation kernel istead of maxpooling # 9
            nn.ReLU(),
            apply_normalization(28),
            nn.Dropout(drop)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            # DepthWise
            nn.Conv2d(in_channels=28, out_channels=28, kernel_size=(3, 3), padding=1, stride=1, groups=28, bias=False), # 11
            #pointwise
            nn.Conv2d(in_channels=28, out_channels=32, kernel_size=(1,1)), # 11
            nn.ReLU(),
            apply_normalization(32),
            nn.Dropout(drop),
            nn.Conv2d(32, 64, (3, 3), padding=1, bias=False), #13
            nn.ReLU(),
            apply_normalization(64),
            nn.Dropout(drop),
            nn.Conv2d(64, 32, (1, 1), stride=1,bias=False), # 13
        )

        self.diated = nn.Sequential(
            nn.Conv2d(32, 30, (3, 3), dilation=2,bias=False), # 21
            nn.ReLU(),
            apply_normalization(30),
            nn.Dropout(drop)
        )

        self.convblock3 = nn.Sequential(
            # DepthWise
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=1, stride=2, groups=30, bias=False), #29
            #pointwise
            nn.Conv2d(in_channels=30, out_channels=32, kernel_size=(1,1)), #29
            nn.ReLU(),
            apply_normalization(32),
            nn.Dropout(drop),
            nn.Conv2d(32, 30, (3, 3), padding=1, bias=False), #33
            nn.ReLU(),
            apply_normalization(30),
            nn.Dropout(drop),
            nn.Conv2d(30, 32, (3, 3), padding=1, bias=False), # 41
            nn.ReLU(),
            apply_normalization(32),
            nn.Dropout(drop),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1, groups=32, bias=False), #49
            #pointwise
            nn.Conv2d(in_channels=32, out_channels=30, padding=1,kernel_size=(1,1)), #49
            nn.ReLU(),
            apply_normalization(30),
            nn.Dropout(drop),
            nn.Conv2d(30, 32, (3, 3), padding=1, bias=False), # 57
            nn.ReLU(),
            apply_normalization(32),
            nn.Dropout(drop),
            nn.Conv2d(32, 20, (3, 3), padding=1, bias=False), #
            nn.ReLU(),
            apply_normalization(20),
            nn.Dropout(drop)
        )


        # Global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(8)
        )

        # Fully connected layer
        self.convblock4 = nn.Sequential(
            nn.Conv2d(20, 10, (1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        #x = self.trans2(x)
        x = self.diated(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x = self.convblock4(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)