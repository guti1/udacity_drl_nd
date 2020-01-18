import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,
                 layer1_n=512, layer2_n=256, layer3_n=128,
                 layer4_n=64, layer5_n=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layer[1...5]_n: Number of neurons in layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        #Fully connected layers....

        self.fcl1 = nn.Linear(state_size, layer1_n)
        self.fcl2 = nn.Linear(layer1_n, layer2_n)
        self.fcl3 = nn.Linear(layer2_n, layer3_n)
        self.fcl4 = nn.Linear(layer3_n, layer4_n)
        self.fcl5 = nn.Linear(layer4_n, layer5_n)
        # output layer ending in actions
        self.ol = nn.Linear(layer5_n, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        # defining the forwad pass (flow from state to actions)
        x = F.selu(self.fcl1(state))
        x = F.selu(self.fcl2(x))
        x = F.selu(self.fcl3(x))
        x = F.selu(self.fcl4(x))
        x = F.selu(self.fcl5(x))
        x = self.ol(x)

        return x