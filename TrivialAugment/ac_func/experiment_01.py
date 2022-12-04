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


class Func_02(nn.Module):
    '''
    write something meaningful here
    '''

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return self.beta_mix * (-1 * (torch.cos(input) * self.beta)) + (1 - self.beta_mix) * torch.atan(input)


class Func_03(nn.Module):
    '''
    write something meaningful here
    '''

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.log1p(torch.exp(torch.sigmoid(input) * torch.abs(input))) * torch.minimum(input, torch.zeros_like(input))


class Func_04(nn.Module):
    '''
    write something meaningful here
    '''

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.maximum(torch.maximum(torch.sigmoid(input) * torch.sigmoid(input), torch.zeros_like(input)), torch.asinh(input))
