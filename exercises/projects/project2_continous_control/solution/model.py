import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, use_batch_norm, use_xavier_init):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden layers([int]): list of the sizes each hidden layers
            use_batch_norm (bool): use or not batch norm between layers
            use_xavier_init (bool): use Xavier weight init instead of original
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.batch_norm = use_batch_norm

        def _init_weights_opt(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                fan_in = m.weight.data.size()[0]
                lim = 1. / np.sqrt(fan_in)
                nn.init.uniform_(m.weight, -lim, lim)

        # Construction of the NN conditionally on batch-norm
        if self.batch_norm:
            layers = [state_size] + hidden_layers
            # Input layer
            self.hidden_layers = nn.ModuleList([nn.BatchNorm1d(state_size)])
            # Remaining hidden layers
            for i in range(len(layers) - 1):
                self.hidden_layers.extend([nn.Linear(layers[i], layers[i + 1])])
                self.hidden_layers.extend([nn.BatchNorm1d(layers[i + 1])])

        # Construction without batch-norm
        else:
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(state_size, hidden_layers[0])]
            )
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(n1, n2) for n1, n2 in layer_sizes])

        # Finally the output layer
        self.output = nn.Linear(hidden_layers[-1], action_size)

        # Weight init
        if not use_xavier_init:
            self.hidden_layers.apply(_init_weights)
            self.output.weight.data.uniform_(-3e-3, 3e-3)

        else:
            self.hidden_layers.apply(_init_weights_opt)
            self.output.apply(_init_weights_opt)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions.
            Note the final tanh activation which is mapping the actions to [-1, 1]
        """
        x = state

        if self.batch_norm:
            x = self.hidden_layers[0](x)
            for i in range(1, len(self.hidden_layers) - 1, 2):
                x = F.relu(self.hidden_layers[i + 1](self.hidden_layers[i](x)))
        else:
            for j in self.hidden_layers:
                x = F.relu(j(x))

        return torch.tanh(self.output(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, use_xavier_init):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden layers([int]): list of the sizes each hidden layers
            use_xavier_init(bool): use Xavier weight init instead of original

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        def _init_weights_opt(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                fan_in = m.weight.data.size()[0]
                lim = 1. / np.sqrt(fan_in)
                nn.init.uniform_(m.weight, -lim, lim)

        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(state_size, hidden_layers[0]),
                nn.Linear(hidden_layers[0] + action_size, hidden_layers[1]),
            ]
        )
        layer_sizes = zip(hidden_layers[1:-1], hidden_layers[2:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # Finally the output layer
        self.output = nn.Linear(hidden_layers[-1], 1)

        # Weight init
        if not use_xavier_init:
            self.hidden_layers.apply(_init_weights)
            self.output.weight.data.uniform_(-3e-4, 3e-4)

        else:
            self.hidden_layers.apply(_init_weights_opt)
            self.output.apply(_init_weights_opt)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = F.relu(self.hidden_layers[0](state))
        x = torch.cat((x, action), dim=1)

        for i in range(1, len(self.hidden_layers)):
            x = F.relu(self.hidden_layers[i](x))

        return self.output(x)
