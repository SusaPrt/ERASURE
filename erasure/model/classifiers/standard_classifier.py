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
    def __init__(self, n_classes, inputsize=(6,128), hidden_size=64, kernel_size=6):
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

        return intermediate_output, out
    
  # PAMAP2 - 18 different physical activities, performed by 9 subjects wearing 3 inertial measurement units and a heart rate monitor.
  # realWorld2016 contains 15 subjects. 8 Activities: 0: climbingdown 1: climbingup 2: jumping 3: lying 4: running 5: sitting 6: standing 7: walking, 7 positions: chest, forearm, head, shin, thigh, upper arm, and waist.

class DeepConvLSTM(nn.Module):    
    def __init__(self, n_classes, train_on_gpu, n_hidden=128, n_layers=1, n_filters=64, filter_size=5, drop_prob=0.5, NB_SENSOR_CHANNELS=6, SLIDING_WINDOW_LENGTH=128):
        super(DeepConvLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.NB_SENSOR_CHANNELS = NB_SENSOR_CHANNELS
        self.SLIDING_WINDOW_LENGTH = SLIDING_WINDOW_LENGTH
        self.train_on_gpu = train_on_gpu

        self.conv1 = nn.Conv1d(self.NB_SENSOR_CHANNELS, n_filters, self.filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, self.filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, self.filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, self.filter_size)

        self.lstm1  = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        self.lstm2  = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        
        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):

        batch_size = x.size(0)  
        hidden = self.init_hidden(batch_size)
        
        x = x.view(-1, self.NB_SENSOR_CHANNELS, self.SLIDING_WINDOW_LENGTH)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(batch_size, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)
        
        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)
        
        out = x.view(batch_size, -1, self.n_classes)[:,-1,:]
        
        return hidden,out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden