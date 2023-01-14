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
    run_resnetLastHPO/cifar10_DrNASOptimizer_35_128_0.003_10_0.0025_0.001_10_0.1_5.0/1/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(torch.log(2 * torch.exp(input) + 0.1)) * torch.relu(input)


class Func_02(nn.Module):
    """
    run_resnetLastHPO/cifar10_DrNASOptimizer_35_128_0.0003_10_0.025_0.01_10_0.1_0.5/1/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(torch.abs(torch.exp(input) + torch.relu(input))) * torch.relu(input)


class Func_03(nn.Module):
    """
    cifar10_DrNASOptimizer_10_128_0.003_10_0.0025_0.001_10_0.1_5.0/3/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(torch.sqrt(torch.relu(input) + torch.relu(input))) * torch.relu(input)


class Func_04(nn.Module):
    """
    cifar10_DARTSOptimizer_10_128_0.0003_10_0.0025_0.001_10_0.1_0.5/1/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.sigmoid(torch.sinc(torch.exp(-torch.pow(input, 2)) * torch.sin(input))) * torch.relu(input)

class Func_05(nn.Module):
    """
    cifar10_DrNASOptimizer_10_128_0.0003_10_0.0025_0.001_10_0.1_0.5/0/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5
    def forward(self, input):
        return torch.sigmoid(torch.sqrt(torch.relu(input) + torch.sqrt(input.clamp(min=0.01)))) * torch.relu(input)


class Func_06(nn.Module):
    """
    cifar10_DrNASOptimizer_10_128_0.003_10_0.0025_0.001_10_0.1_5.0/0/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5
    def forward(self, input):
        return torch.sigmoid(torch.relu(torch.pow(input, 3) + torch.relu(input))) * torch.relu(input)


class Func_07(nn.Module):
    """
    cifar10_DARTSOptimizer_10_128_0.003_10_0.0025_0.001_10_0.1_0.5/1/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5
    def forward(self, input):
        # torch.minimum(torch.sinh(torch.sinh(input) * torch.exp(input)), torch.relu(input))
        x_clamp = torch.where(torch.abs(input) > 2, torch.sign(input) * 2, input)
        return torch.minimum(torch.sinh(torch.sinh(x_clamp) * torch.exp(x_clamp)), torch.relu(input))


class Func_08(nn.Module):
    """
    cifar10_DARTSOptimizer_10_128_0.003_10_0.0025_0.001_10_0.1_0.5/4/log.log
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5
    def forward(self, input):
        return torch.minimum(torch.exp(2 * torch.relu(input)), torch.relu(input))


class Softplus(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.where(input < 20, torch.log1p(torch.exp(input)), input)