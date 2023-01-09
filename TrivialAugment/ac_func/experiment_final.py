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
    run_NewACell/cifar10_DrNASOptimizer_15_128_0.0003_10_0.025_0.001_10_0.1_5.0/12/log.log
    12
    """

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(torch.abs(torch.relu(input)-torch.tanh(input))) * -torch.relu(-input)


class Func_02(nn.Module):
    """
    cifar10_DARTSOptimizer_5_128_0.0003_10_0.025_0.001_10_0.1_5.0
    5
    --> just relu
    """

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(torch.zeros_like(input)) * torch.relu(input)


class Func_03(nn.Module):
    """
    (cifar10_DARTSOptimizer_5_128_0.0003_10_0.025_0.001_10_0.1_5.0
    3
    """

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.maximum(torch.asinh(torch.asinh(input)), torch.relu(input))


class Func_04(nn.Module):
    """
    clamp 100
    run_NewACell2/cifar10_GDASOptimizer_15_128_0.0003_100_0.025_0.001_10_0.01_5.0/11/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.sigmoid(
            torch.relu(torch.div(1.0, input).clamp(max=100, min=-100) + 1) * torch.relu(input))


class Func_05(nn.Module):
    """
    clamp 100
    run_NewACell2/cifar10_DrNASOptimizer_15_128_0.0003_100_0.025_0.001_10_0.1_5.0/3/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5
    def forward(self, input):
        a = torch.asinh(input)
        b = torch.div(1.0, input).clamp(max=100, min=-100)
        c = torch.abs(input)

        a = torch.maximum(a, b)
        a = torch.div(1.0, a).clamp(max=100, min=-100)
        a = torch.sigmoid(a)
        res = a * c

        return res
