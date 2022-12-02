import torch
import torch.nn as nn


class Func_01(nn.Module):
    '''
    write something meaningful here
    '''

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.sigmoid(torch.sqrt(
            (self.beta_mix * torch.abs(input) + (1 - self.beta_mix) * torch.sinc(input)).clamp(
                min=1e-05))) * torch.relu(input)
