import torch
from torch import nn
from torch.nn import functional as F


class Prompt(nn.Module):
    def __init__(self, prompt_vectors):
        super(Prompt, self).__init__()
        self.vars = nn.ParameterList()
        w1 = nn.Parameter(prompt_vectors)
        self.vars.append(w1)

    def forward(self, x, vars=None):
        if vars == None:
            vars = self.vars
        x = torch.matmul(x, vars[0])
        # x = F.linear(x, vars[0], vars[1])
        # x = torch.relu(x)
        # x = F.linear(x, vars[2], vars[3])
        # x = torch.relu(x)
        return x

    def parameters(self):
        return self.vars
