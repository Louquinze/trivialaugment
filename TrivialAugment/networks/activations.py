from TrivialAugment.networks.unary_operator import *
from TrivialAugment.networks.binary_operator import *


class test_ac(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)