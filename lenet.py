from torch import nn, zeros
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#замена MaxPooling на Conv2d+stride=2, 
#замена 1х5х5 фильтра на 2х3х3+padding=1,
#1 FC слой
class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6, 6, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            

            nn.Conv2d(6, 16, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(256, 10))

    def forward(self, x):
        x = self.conv(x)
        #print(x.size())
        x = x.view(-1,256)
        #print(x.size())
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#dropout
class LeNet2(nn.Module):
    def __init__(self):
        super(LeNet2, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4))
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#batch normalization
class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16))
        self.fc = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(120),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(84),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class SuperLeNet(nn.Module):
    def __init__(self):
        super(SuperLeNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4))
        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128,10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class LeNet32(nn.Module):
    def __init__(self):
        super(LeNet32, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Linear(1024, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class LeNetEnsemble(nn.Module):
    def __init__(self, num, device):
        super(LeNetEnsemble,self).__init__()
        self.models = nn.ModuleList([SuperLeNet().to(device) for _ in range(num)])
        self.device = device
        
    def forward(self, x):
        output = zeros([x.size(0), 10]).to(self.device)
        for model in self.models:
            output += model(x)
        return output
