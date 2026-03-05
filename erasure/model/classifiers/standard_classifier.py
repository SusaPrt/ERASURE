import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisNN(nn.Module):
    def __init__(self, n_classes):
        super(IrisNN, self).__init__()
        
        self.fc1 = nn.Linear(4, 50)  
        self.fc2 = nn.Linear(50, 30)          
        self.fc3 = nn.Linear(30, n_classes) 
        
        self.relu = nn.ReLU() 
        self.flatten = nn.Flatten()
        self.last_layer = self.fc3

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        intermediate_output = x   
        x = self.fc3(x)    
        x = x.squeeze(1)        
        return intermediate_output, x

class AdultNN(nn.Module):
    def __init__(self, n_classes, n_layers=3, inputsize=68, hidden_size=100):
        super(AdultNN, self).__init__()
        # pass n_layers, inputsize, hidden_size
        self.fc1 = nn.Linear(inputsize, hidden_size)  
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers-1)])
        self.fc3 = nn.Linear(hidden_size, n_classes) 
        
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        intermediate_output = x   
        x = self.fc3(x)    
        x = x.squeeze(1)        
        return intermediate_output, x
    
class SpotifyNN(nn.Module):
    def __init__(self, n_classes, n_layers=3, inputsize=15, hidden_size=100):
        super(SpotifyNN, self).__init__()
        print("The number of classes is: ", n_classes)
        self.fc1 = nn.Linear(inputsize, hidden_size)  
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers-1)])
        self.fc3 = nn.Linear(hidden_size, n_classes)
        
        self.relu = nn.ReLU() 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        intermediate_output = x
        x = self.fc3(x)
        return intermediate_output, x

class HAR_NN(nn.Module):
    def __init__(self, n_classes, inputsize=(6,128)):
        super(HAR_NN, self).__init__()
        
        channels, seq_len = inputsize
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=5)
        self.bn1   = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.bn2   = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        self.bn3   = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)

        self._to_linear = None
        self._get_conv_output(inputsize)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self._to_linear = x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)

        intermediate_output = x

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return intermediate_output, x
"""    def __init__(self, n_classes, inputsize=(6,128), hidden_size=64, kernel_size=6):
        super(HAR_NN, self).__init__()
        in_channels = inputsize[0]

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(output_size=28)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 28, out_features=500),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=500, out_features=250),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=250, out_features=n_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)

        intermediate_output = out

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return intermediate_output, out"""