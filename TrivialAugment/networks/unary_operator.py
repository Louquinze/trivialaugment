import torch
import torch.nn as nn


class Identitiy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Sign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -x


class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class Pow2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(x, 2)


class Pow3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x = torch.where(torch.abs(x) > self.inf ** (1 / 3), torch.sign(x) * self.inf ** (1 / 3), x)
        x = torch.pow(x, 3)
        return x


class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.where(x < 1e-3, 1e-3, x)
        return torch.sqrt(x)


class BetaMul(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.beta

    def reset_beta(self):
        self.beta = nn.Parameter(torch.ones(1))


class BetaAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.beta

    def reset_beta(self):
        self.beta = nn.Parameter(torch.ones(1))


class Beta(nn.Module):
    def __init__(self, gdas=False):
        super().__init__()
        self.gdas = gdas
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.zeros_like(x) + self.beta

    def reset_beta(self):
        self.beta = nn.Parameter(torch.ones(1))


class LogAbsEps(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-4

    def forward(self, x):
        return torch.log(torch.abs(x) + self.eps)


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Cos(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


class Sinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sinh(x)


class Cosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cosh(x)


class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)


class Asinh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


class Atan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.atan(x)


class Sinc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sinc(x)


class Max0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


class Min0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return -torch.relu(-x)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)


class LogExp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x < 4, torch.log1p(torch.exp(x)), x)


class Exp2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.where(torch.abs(x) < torch.sqrt(self.inf), torch.sqrt(self.inf), x)
        x = torch.exp(-torch.pow(x, 2))
        return x


class Erf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.erf(x)
        return x
