import torch
from torch import nn
from torch.nn import functional as F


class Final_m(nn.Module):
    def __init__(self, gnn, prompt):
        super(Final_m, self).__init__()
        self.vars = nn.ParameterList()
        self.gnn = gnn
        self.prompt = prompt
        # for name, param in gnn.named_parameters():
        #     print('name', name)
        #     self.vars.append(param)
        for param in self.gnn.parameters():
            self.vars.append(param)
        for param in self.prompt.parameters():
            self.vars.append(param)

    def forward(self, x, adjs):
        out = self.gnn(x, adjs)
        res = self.prompt(out)
        return res

    def parameters(self):
        return self.vars
