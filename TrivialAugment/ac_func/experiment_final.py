import torch
import torch.nn as nn


def logexp(input):
    """
    Ln(1+e^x)

    for fixing inf issues form x = 10 ~ ln(32000) --> linear function

    """

    return input.where(input > 10, torch.log1p(torch.exp(input)))


class Func_01(nn.Module):
    """
    run_final/cifar10_EdgePopUpOptimizer_1_128_0.0003_32768_0.025_0.001_10_0.1_5.0/69/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_1 = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)

    def forward(self, input):
        return torch.sigmoid(
            torch.sinc(torch.exp(
                -self.beta_1 * torch.pow(logexp(input) - torch.tanh(input), 2).clamp(max=32768, min=-32768)).clamp(
                max=32768, min=-32768))) * torch.relu(input)


class Func_02(nn.Module):
    """
    (85.33, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/80')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_1 = nn.Parameter(torch.ones((channels, 1, 1)))
        self.beta_2 = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        self.beta_3 = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)

    def forward(self, input):
        return self.beta_3 * torch.pow(torch.exp(-self.beta_2 * torch.abs(-torch.relu(-input) - self.beta_1)), 2) + (
                1 - self.beta_3) * torch.asinh(input)


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
        return torch.sigmoid(torch.minimum(torch.log(torch.abs(input) + 1e-05),
                                           torch.sinh(input).clamp(max=32768, min=-32768))) * torch.sqrt(
            input.clamp(min=1e-05))


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
        self.beta_1 = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.sigmoid(self.beta_1) * torch.relu(input)
