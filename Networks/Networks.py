import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import torchvision.transforms as T

HIDDEN_LAYER = 128
HIDDEN_LAYER2 = 64

class Network(nn.Module):
    def __init__(self,state_apace_n, action_space_n):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_apace_n,HIDDEN_LAYER)
        self.fc2 = nn.Linear(HIDDEN_LAYER, HIDDEN_LAYER2)
        self.fc3 = nn.Linear(HIDDEN_LAYER2, action_space_n)

    def forward(self, x):
        if type(x)!=float:
            x = x.view(-1,1).to(device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x