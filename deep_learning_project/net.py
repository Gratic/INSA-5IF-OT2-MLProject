from turtle import forward
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionNetwork(nn.Module):
    def __init__(self):
        super(LinearRegressionNetwork, self).__init__()
        self.fc1 = nn.Linear(36*36, 2)
    
    def forward(self, x):
        x = x.view(-1, 36*36)
        x = self.fc1(x)
        return x

class FirstNeuralNetwork(nn.Module):
    def __init__(self):
        super(FirstNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SecondNeuralNetwork(nn.Module):
    def __init__(self):
        super(SecondNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 48, 5)
        self.fc1 = nn.Linear(4800, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = x.view(-1, 4800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ThirdNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(ThirdNeuralNetwork, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 72, 3)
        self.conv2 = nn.Conv2d(72, 144, 3)
        self.fc1 = nn.Conv2d(144, 256, 7)
        self.fc2 = nn.Conv2d(256, 2, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FourthNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(FourthNeuralNetwork, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Conv2d(64, 64, 7)
        self.fc2 = nn.Conv2d(64, 2, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FifthNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(FifthNeuralNetwork, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Conv2d(256, 256, 6)
        self.fc2 = nn.Conv2d(256, 512, 1)
        self.fc3 = nn.Conv2d(512, 2, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class SixthNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(SixthNeuralNetwork, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 512, 3)
        self.fc1 = nn.Conv2d(512, 512, 1)
        self.fc2 = nn.Conv2d(512, 256, 1)
        self.fc3 = nn.Conv2d(256, 2, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x