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
    (86.35, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/13')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
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
        return logexp(torch.sigmoid(input) * logexp(input)) * -torch.relu(-input)


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


class Func_05(nn.Module):
    """
    (85.04, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/94')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.cos(torch.sqrt(input.clamp(min=1e-05)) * logexp(input)) - logexp(input)


class Func_06(nn.Module):
    """
    (84.27, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/34')
    (activation_cell-edge(1,2)): Abs_op()
    (activation_cell-edge(1,3)): Log()  -> cancels out
    (activation_cell-edge(1,8)): Asinh()
    (activation_cell-edge(4,5)): Maximum()
    (activation_cell-edge(6,7)): Atan()
    (activation_cell-edge(9,10)): BetaMix()
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        # self.beta_mix = 0.5

    def forward(self, input):
        return self.beta_mix * torch.atan(torch.abs(input)) + (1 - self.beta_mix) * torch.asinh(input)


class Func_07(nn.Module):
    """
    (83.88, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/57')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.sigmoid(torch.minimum(torch.sigmoid(input), input) + self.beta) * torch.abs(input)


class Func_08(nn.Module):
    """
    (83.84, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/91')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_mix = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        self.beta_sub = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return self.beta_mix * -torch.exp(-self.beta_sub * torch.abs(torch.erf(input) - (input + self.beta))) + (
                    1 - self.beta_mix) * torch.erf(input)


class Func_09(nn.Module):
    """
    (83.65, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/72')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.exp2(torch.sigmoid(torch.atan(input)) * torch.erf(input)) - (input * self.beta)


class Func_10(nn.Module):
    """
    (83.58, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/21')

    this is just a -1 * relu(x + 1) + 1

    (activation_cell-edge(1,2)): Abs_op()  --> canceled out min(|x|, relu(x)) = relu(x)
    (activation_cell-edge(1,3)): Maximum0()
    (activation_cell-edge(1,8)): Sign()
    (activation_cell-edge(4,5)): Minimum()
    (activation_cell-edge(6,7)): Cos()
    (activation_cell-edge(9,10)): Minimum()
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        # self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.minimum(torch.cos(torch.relu(input)), -input)


class Func_11(nn.Module):
    """
    (83.49, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/78')
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones((channels, 1, 1)))
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.sigmoid(torch.cos(torch.log(input.clamp(min=1e-05)) * (self.beta + input))) * torch.tanh(input)


class Func_12(nn.Module):
    """
    (83.36, '../save_only_topk/run_topk/cifar10/DARTSTopKOptimizer/10/512/83')

    (activation_cell-edge(1,2)): Power() # 2
    (activation_cell-edge(1,3)): Cosh()
    (activation_cell-edge(1,8)): Identity()
    (activation_cell-edge(4,5)): Mul()
    (activation_cell-edge(6,7)): Sigmoid()
    (activation_cell-edge(9,10)): ExpBetaSubAbs()
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        self.beta_sub = nn.Parameter(torch.ones((channels, 1, 1)) - 0.5)
        # self.beta_mix = 0.5

    def forward(self, input):
        return torch.exp(-self.beta_sub * torch.abs(torch.sigmoid(torch.pow(input, 2) * torch.cosh(input)) - input))
