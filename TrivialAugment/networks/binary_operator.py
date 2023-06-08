import torch
import torch.nn as nn


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0] + x[1]


class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0] * x[1]


class Sub(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0] - x[1]


class Div(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.where(torch.abs(x[0]) > (self.inf / 10), torch.sign(x[0]) * (10 / self.inf), x[0]) / torch.where(
            torch.abs(x[1]) < 0.1, torch.sign(x[1]) * 0.1, x[1])
        x = torch.clamp(x, max=self.inf, min=-self.inf)
        x = torch.nan_to_num(x, posinf=self.inf, neginf=-self.inf)
        return x


class Maximum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(x[0], x[1])


class Minimum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.minimum(x[0], x[1])


class SigMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x[0]) * x[1]


class BetaAbsExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = torch.abs(x[0] - x[1])
        x = torch.exp(-torch.relu(self.beta) * x)
        x = torch.clamp(x, max=self.inf, min=-self.inf)
        return x

    def reset_beta(self):
        self.beta = nn.Parameter(torch.ones(1))


class BetaPow2Exp(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = torch.pow(x[0] - x[1], 2)
        x = torch.clamp(x, max=self.inf, min=-self.inf)
        x = torch.exp(-torch.relu(self.beta) * x)
        return x

    def reset_beta(self):
        self.beta = nn.Parameter(torch.ones(1))


class BetaMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = torch.sigmoid(self.beta) * x[0] + (1 - torch.sigmoid(self.beta)) * x[1]
        return x

    def reset_beta(self):
        self.beta = nn.Parameter(torch.zeros(1))


class Left(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


class Right(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[1]
