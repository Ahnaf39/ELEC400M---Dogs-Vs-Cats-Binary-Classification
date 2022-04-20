import torch
from torch import nn
from torch.nn import functional as F
class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()
            
        # define 2 convolutional layers
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size = 2)

        # define linear layers to project 
        self.linear0 = nn.Linear(512 * (256 // 16 - 4) * (256 // 16 - 4), 4096)
        self.linear1 = nn.Linear(4096, 10)
    

    def forward(self, x):
        #use relu as activation layer, use max pooling to compress data and take only the strongest signal
        output = F.relu(self.conv0(x))
        output = F.relu(self.conv1(output))
        output = self.max_pool(output)
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = self.max_pool(output)
        output = F.relu(self.conv4(output))
        output = F.relu(self.conv5(output))
        output = self.max_pool(output)
        output = F.relu(self.conv6(output))
        output = F.relu(self.conv7(output))
        output = self.max_pool(output)
        
        # flatten the feature map except the batch dim
        output = torch.flatten(output, 1)

        output = F.relu(self.linear0(output))
        output = F.dropout(output, 0.5)
        output = self.linear1(output)

        return output