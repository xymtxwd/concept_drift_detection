import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class Generator_u2m(nn.Module):
    def __init__(self, outdim=500):
        super(Generator_u2m, self).__init__()
        self.conv1 = Conv_Block(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = Conv_Block(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout()
        self.fc = Dense_Block(800, outdim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x
    
class decoder(nn.Module):
    def __init__(self, task='u2m', outdim=500):
        super(decoder, self).__init__()
        self.fc = Dense_Block(outdim, 800)
        self.layer = nn.Sequential(
                        nn.ConvTranspose2d(50,20,13,stride=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(20),
                        nn.ConvTranspose2d(20,1,13,stride=1),
                        nn.ReLU())
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 50, 4, 4)
        x = self.layer(x)
        return x

class Classifier_u2m(nn.Module):
    def __init__(self, n_output, outdim=500):
        super(Classifier_u2m, self).__init__()
        self.fc = nn.Linear(outdim, n_output)

    def forward(self, x):
        x = self.fc(x)
        return x


class Net_f(nn.Module):
    def __init__(self, task='s2m', outdim=500):
        super(Net_f, self).__init__()
        if task == 's2m':
            self.generator = Generator_s2m()
        elif task =='u2m' or task == 'm2u':
            self.generator = Generator_u2m(outdim=outdim)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, adaption = False):
        x = self.generator(x)
        return x

class Net_c_cway(nn.Module):
    def __init__(self, task='s2m', outdim=500):
        super(Net_c_cway, self).__init__()
        if task =='u2m' or task == 'm2u':
            self.classifier = Classifier_u2m(10, outdim=outdim)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.classifier(x)
        return x
