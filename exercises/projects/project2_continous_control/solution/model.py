import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _reset_parameters(layers, final_weight_lim):
    for layer in layers:
        if layer != layers[-1]:
            fan_in = layer.weight.data.size()[0]
            lim = 1.0 / np.sqrt(fan_in)
            nn.init.uniform_(layer.weight, -lim, lim)
        else:
            layer.weight.data.uniform_(-final_weight_lim, final_weight_lim)


def _reset_parameters_xavier(layers):
    for layer in layers:
        layer.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        hidden_layers,
        use_batch_norm,
        use_xavier_init,
    ):
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

        # Construction of the NN conditionally on batch-norm
        if self.batch_norm:
            layers = [state_size] + hidden_layers
            # Input layer
            self.hidden_layers = nn.ModuleList()
            # Remaining hidden layers
            for i in range(len(layers) - 1):
                self.hidden_layers.extend(
                    [nn.Linear(layers[i], layers[i + 1], bias=False)]
                )
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

        if use_xavier_init:
            _reset_parameters_xavier(self.hidden_layers)
        else:
            _reset_parameters(self.hidden_layers, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions.
            Note the final tanh activation which is mapping the actions to [-1, 1]
        """
        x = state

        if self.batch_norm:
            x = self.hidden_layers[0](x)
            for i in range(1, len(self.hidden_layers) - 1, 2):
                x = F.leaky_relu(self.hidden_layers[i + 1](self.hidden_layers[i](x)))
        else:
            for j in self.hidden_layers:
                x = F.leaky_relu(j(x))

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

        if use_xavier_init:
            _reset_parameters_xavier(self.hidden_layers)
        else:
            _reset_parameters(self.hidden_layers, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = F.leaky_relu(self.hidden_layers[0](state))
        x = torch.cat((x, action), dim=1)

        for i in range(1, len(self.hidden_layers)):
            x = F.leaky_relu(self.hidden_layers[i](x))

        return self.output(x)
