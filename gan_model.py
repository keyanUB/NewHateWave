import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

# Generator model
class Generator(nn.Module):
    # the input dim is the length of the attribute vector
    # the output dim corresponds to the size of the sentence embedding.
    def __init__(self, input_len: int):
        super(Generator, self).__init__()
        self.inpuyt_layer = nn.Linear(input_len, 100)
        self.hidden_fc0 = nn.Linear(100, 400)
        self.hidden_fc1 = nn.Linear(400, 900)
        self.hidden_fc2 = nn.Linear(900, 1500)
        self.hidden_fc3 = nn.Linear(1500, 1500)
        self.hidden_fc4 = nn.Linear(1500, 1200)
        self.hidden_fc5 = nn.Linear(1200, 1000)
        self.hidden_fc6 = nn.Linear(1000, 900)
        self.dpout = nn.Dropout(p=0.2)
        self.hidden_fc7 = nn.Linear(900, 768)
        

    def forward(self, x):
        h_1 = F.relu(self.inpuyt_layer(x))
        h_2 = F.relu(self.hidden_fc0(h_1))
        h_3 = F.relu(self.hidden_fc1(h_2))
        h_4 = F.relu(self.hidden_fc2(h_3))
        h_5 = F.relu(self.hidden_fc3(h_4))
        h_6 = F.relu(self.hidden_fc4(h_5))
        h_7 = F.relu(self.hidden_fc5(h_6))
        h_8 = F.relu(self.hidden_fc6(h_7))
        h_9 = self.dpout(h_8)
        h_10 = self.hidden_fc7(h_9)

        output = h_10
        
        return output

# Discriminator model
class Discriminator(nn.Module):
    # the inpt dim equals to the size of the sentence embedding.
    # the output dim is 1 to check if it's hateful or not.
    def __init__(self, input_dim: int):
        super(Discriminator, self).__init__()
        self.inpuyt_layer = nn.Linear(input_dim, 1000)
        self.hidden_fc0 = nn.Linear(1000, 768)
        self.hidden_fc1 = nn.Linear(768, 400)
        self.hidden_fc2 = nn.Linear(400, 100)
        self.hidden_fc3 = nn.Linear(100, 50)
        self.hidden_fc4 = nn.Linear(50, 1)

    def forward(self, x):
        h_1 = F.relu(self.inpuyt_layer(x))
        h_2 = F.relu(self.hidden_fc0(h_1))
        h_3 = F.relu(self.hidden_fc1(h_2))
        h_4 = F.relu(self.hidden_fc2(h_3))
        h_5 = F.relu(self.hidden_fc3(h_4))

        output = torch.sigmoid(self.hidden_fc4(h_5))
        
        return output