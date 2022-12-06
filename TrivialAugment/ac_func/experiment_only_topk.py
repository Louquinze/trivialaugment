import torch
import torch.nn as nn


def logexp(input):
    """
    Ln(1+e^x)

    for fixing inf issues form x = 10.37 ~ ln(32000) --> linear function

    """

    return torch.where(input < 10.37, torch.log1p(torch.exp(input)), input)


class Func_01(nn.Module):
    """
    (86.35, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/13')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.asinh(torch.sigmoid(torch.sigmoid(input)) * torch.relu(input)) * self.beta


class Func_02(nn.Module):
    """
    (85.33, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/80')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        # todo check if -relu(-x) == min0
        return torch.minimum(torch.sigmoid(-input - torch.cosh(input)), -torch.relu(-input))


class Func_03(nn.Module):
    """
    (85.06, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/18')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        # todo check if -relu(-x) == min0
        return torch.log1p(torch.exp(torch.sigmoid(input) * torch.log1p(torch.exp(input)))) * -torch.relu(-input)


class Func_04(nn.Module):
    """
    (85.04, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/77')
    (activation_cell-edge(1,2)): Sinh()
    activation_cell-edge(1,3)): Cosh()
    (activation_cell-edge(1,8)): Asinh()
    (activation_cell-edge(4,5)): Maximum()
    (activation_cell-edge(6,7)): Zero (stride=1)
    (activation_cell-edge(9,10)): Maximum()
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.maximum(torch.zeros_like(input), torch.asinh(input))


class Func_04(nn.Module):
    """
    (85.04, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/94')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.cos(torch.sqrt(input.clamp(min=1e-05)) * torch.log1p(torch.exp(input))) - torch.log1p(
            torch.exp(input))
