import torch
import torch.nn as nn


class LeNet(nn.Module):
   # def __init__(self):     
   #     super(LeNet, self).__init__()
   #     # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
   #     self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
   #     # Max-pooling
   #     self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
   #     # Convolution
   #     self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
   #     # Max-pooling
   #     self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
   #     # Fully connected layer
   #     self.fc1 = nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
   #     self.fc2 = nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
   #     self.fc3 = nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)
   #     
   # def forward(self, x):
   #     # convolve, then perform ReLU non-linearity
   #     x = torch.nn.functional.relu(self.conv1(x))  
   #     # max-pooling with 2x2 grid 
   #     x = self.max_pool_1(x) 
   #     # convolve, then perform ReLU non-linearity
   #     x = torch.nn.functional.relu(self.conv2(x))
   #     # max-pooling with 2x2 grid
   #     x = self.max_pool_2(x)
   #     # first flatten 'max_pool_2_out' to contain 16*5*5 columns
   #     # read through https://stackoverflow.com/a/42482819/7551231
   #     x = x.view(-1, 16*5*5)
   #     # FC-1, then perform ReLU non-linearity
   #     x = torch.nn.functional.relu(self.fc1(x))
   #     # FC-2, then perform ReLU non-linearity
   #     x = torch.nn.functional.relu(self.fc2(x))
   #     # FC-3
   #     x = torch.nn.functional.softmax(self.fc3(x))
   #     
   #     return x
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 *5 *5,120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84,10),
            nn.Softmax()
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


