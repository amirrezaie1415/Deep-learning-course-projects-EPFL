import torch
from torch import nn
from torch.nn import functional as F

        
class ConvNet(nn.Module):
    """This is the version of the neural network without weight sharing. """
    def __init__(self, nb_channels=32, auxiliary_loss=False):
        super().__init__()
        self.auxiliary_loss = auxiliary_loss
        self.conv1 = nn.Sequential(nn.Conv2d(2, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.Dropout2d(p=0.1))
        
        self.conv2 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.Dropout2d(p=0.1)) 
        
        self.conv3 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.Dropout2d(p=0.1))
        
        self.conv4 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.Dropout2d(p=0.1))
        
        self.conv5 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.Dropout2d(p=0.1))
        
        self.conv6 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.Dropout2d(p=0.1))
        
        self.fc1 = nn.Sequential(nn.Linear(nb_channels * 2 * 2, 20),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(p=0.5))
        self.out1 = nn.Linear(20, 1)
        if self.auxiliary_loss:
            self.out2 = nn.Linear(20, 10)
            self.out3 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)        
        out1 = self.out1(x)
        if not self.auxiliary_loss:
            return out1
        else:
            out2 = self.out2(x)
            out3 = self.out3(x)
            return out1, out2, out3

class WeightSharing(nn.Module):
    """This is the version of the neural network with weight sharing. """
    def __init__(self, nb_channels=32, auxiliary_loss=False):
        super().__init__()
        self.auxiliary_loss = auxiliary_loss

        self.conv1 = nn.Sequential(nn.Conv2d(1, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.1))
        
        self.conv2 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.1))

        self.conv3 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.1))

        self.conv4 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.1))

        self.conv5 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.1))
        
        self.conv6 = nn.Sequential(nn.Conv2d(nb_channels, nb_channels, kernel_size=(3, 3), stride=(1, 1), padding=0),
                                   nn.BatchNorm2d(num_features=nb_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.1))

        
        self.fc1 = nn.Sequential(nn.Linear(nb_channels * 2 * 2, 10),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5))
        self.out = nn.Linear(20, 1)
        if self.auxiliary_loss:
            self.aux = nn.Linear(10, 10)
            
    def forward(self, x):
        # split x into the two channels to do the convolution
        x1, x2 = x.chunk(2, dim=1) # split along the 1st dim (the channel one)

        # first image
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)

        x1 = x1.view(x1.size(0), -1)
        x1 = self.fc1(x1)
        
        # second image
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        x2 = self.conv6(x2)

        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc1(x2)
        
        # no need to merge it back since x1 and x2 are views
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        out = self.out(x)
        
        if not self.auxiliary_loss:
            return out
        else:
            aux1 = self.aux(x1)
            aux2 = self.aux(x2)
            return out, aux1, aux2
