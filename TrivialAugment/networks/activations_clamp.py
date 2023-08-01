import torch
import torch.nn as nn

from TrivialAugment.networks.binary_operator_clamp import *
from TrivialAugment.networks.unary_operator_clamp import *


class Func1(nn.Module):
    def __init__(self):
        super(Func1, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.))
        self.beta_m = nn.Parameter(torch.tensor(.5))
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        return torch.sigmoid(self.beta_m) * (torch.pow(x, 2).clamp(max=10) + self.lrelu(x) + self.beta) + (
                    1 - torch.sigmoid(self.beta_m)) * torch.relu(x)


class Func2(nn.Module):
    def __init__(self):
        super(Func2, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return 3*torch.relu(x) + self.beta


class Func3(nn.Module):
    def __init__(self):
        super(Func3, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.))
        self.beta_m = nn.Parameter(torch.tensor(.5))

    def forward(self, x):
        return torch.sigmoid(self.beta_m) * (torch.pow(x, 2).clamp(max=10) + torch.relu(x) + self.beta) + (
                    1 - torch.sigmoid(self.beta_m)) * torch.relu(x)


class f0_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = BetaAdd()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = BetaAdd()
        self.u_4 = Pow3()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow3()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_res18_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_res18_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = nn.ReLU()
        self.u_3 = Tanh()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_res18_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_res18_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ELU()
        self.u_2 = nn.ELU()
        self.u_3 = BetaMul()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_res18_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_res18_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_res18_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_res18_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Identitiy()
        self.u_2 = Identitiy()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_res18_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_res18_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_res18_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_res18_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_res18_cifar100_drnas_0_darts_2_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_res18_cifar100_drnas_0_darts_2_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_res18_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_res18_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_res18_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_res18_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_res18_cifar100_drnas_0_darts_2_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_res18_cifar100_drnas_0_darts_2_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Tanh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_res18_cifar100_drnas_4_darts_0_v3_4_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_res18_cifar100_drnas_4_darts_0_v3_4_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_res18_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_res18_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f20_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f20_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f21_res18_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f21_res18_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f22_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f22_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f23_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f23_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f24_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f24_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f25_res18_cifar100_drnas_2_darts_0_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f25_res18_cifar100_drnas_2_darts_0_v3_0_v4_2, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f26_res18_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f26_res18_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f27_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f27_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f28_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f28_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f29_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f29_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f30_res18_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f30_res18_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Mul()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f31_res18_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f31_res18_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f32_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f32_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f33_res18_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f33_res18_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_wres28x2_cifar10_drnas_0_darts_2_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_wres28x2_cifar10_drnas_0_darts_2_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_wres28x2_cifar10_drnas_0_darts_2_v3_1_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_wres28x2_cifar10_drnas_0_darts_2_v3_1_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Mul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_wres28x2_cifar10_drnas_0_darts_3_v3_2_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_wres28x2_cifar10_drnas_0_darts_3_v3_2_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Sign()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Sqrt()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_wres28x2_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = Sqrt()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_wres28x2_cifar10_drnas_2_darts_0_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_wres28x2_cifar10_drnas_2_darts_0_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_wres28x2_cifar10_drnas_2_darts_0_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_wres28x2_cifar10_drnas_2_darts_0_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Beta()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f20_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f20_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f21_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f21_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f22_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f22_wres28x2_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f23_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f23_wres28x2_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f24_wres28x2_cifar10_drnas_0_darts_2_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f24_wres28x2_cifar10_drnas_0_darts_2_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f25_wres28x2_cifar10_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f25_wres28x2_cifar10_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f26_wres28x2_cifar10_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f26_wres28x2_cifar10_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f27_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f27_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f28_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f28_wres28x2_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f29_wres28x2_cifar10_drnas_0_darts_2_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f29_wres28x2_cifar10_drnas_0_darts_2_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Sqrt()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Asinh()
        self.u_2 = Asinh()
        self.u_3 = Min0()
        self.u_4 = BetaMul()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = Asinh()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = Sub()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = BetaMul()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = BetaMul()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Maximum()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = Min0()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = Sign()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = Sign()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Sub()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = nn.LeakyReLU()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = nn.LeakyReLU()

        self.b_1 = Sub()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Asinh()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Sign()
        self.u_4 = BetaMul()

        self.b_1 = Sub()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_wres28x2_svhn_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_wres28x2_svhn_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = Mul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = Sub()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_wres28x2_svhn_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_wres28x2_svhn_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f20_wres28x2_svhn_drnas_1_darts_3_v3_3_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f20_wres28x2_svhn_drnas_1_darts_3_v3_3_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f21_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f21_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f22_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f22_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Tanh()

        self.b_1 = Maximum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f23_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f23_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Sign()
        self.u_3 = BetaMul()
        self.u_4 = Sign()

        self.b_1 = Sub()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f24_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f24_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Sqrt()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f25_wres28x2_svhn_drnas_0_darts_3_v3_2_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f25_wres28x2_svhn_drnas_0_darts_3_v3_2_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Sqrt()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f26_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f26_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f27_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f27_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Sign()
        self.u_4 = Pow3()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f28_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f28_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Sqrt()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f29_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f29_wres28x2_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.SiLU()
        self.u_2 = nn.SiLU()
        self.u_3 = BetaMul()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f30_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f30_wres28x2_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Sign()
        self.u_2 = BetaMul()
        self.u_3 = Sign()
        self.u_4 = BetaMul()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f31_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f31_wres28x2_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Sign()
        self.u_2 = nn.ReLU()
        self.u_3 = Sign()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f32_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f32_wres28x2_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Sign()
        self.u_2 = Min0()
        self.u_3 = Beta()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = BetaMul()
        self.u_3 = Pow2()
        self.u_4 = nn.ReLU()

        self.b_1 = SigMul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = nn.ReLU()

        self.b_1 = Sub()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = SigMul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_ViTtiny_cifar10_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_ViTtiny_cifar10_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_ViTtiny_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_ViTtiny_cifar10_drnas_3_darts_0_v3_1_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_ViTtiny_cifar10_drnas_3_darts_0_v3_1_v4_2, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.GELU()
        self.u_3 = Sign()
        self.u_4 = nn.GELU()

        self.b_1 = Mul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_ViTtiny_cifar10_drnas_0_darts_2_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_ViTtiny_cifar10_drnas_0_darts_2_v3_2_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.GELU()
        self.u_3 = Sign()
        self.u_4 = nn.LeakyReLU()

        self.b_1 = Sub()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Beta()
        self.u_4 = Exp()

        self.b_1 = SigMul()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU()
        self.u_4 = nn.LeakyReLU()

        self.b_1 = SigMul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = nn.ReLU()

        self.b_1 = Add()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_ViTtiny_cifar10_drnas_3_darts_0_v3_0_v4_3(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_ViTtiny_cifar10_drnas_3_darts_0_v3_0_v4_3, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f20_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f20_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Identitiy()

        self.b_1 = SigMul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f21_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f21_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f22_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f22_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow3()
        self.u_4 = nn.ReLU()

        self.b_1 = Mul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f23_ViTtiny_cifar10_drnas_0_darts_2_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f23_ViTtiny_cifar10_drnas_0_darts_2_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Sign()
        self.u_4 = Identitiy()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f24_ViTtiny_cifar10_drnas_0_darts_2_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f24_ViTtiny_cifar10_drnas_0_darts_2_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow3()
        self.u_3 = Pow2()
        self.u_4 = Identitiy()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f25_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f25_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow3()
        self.u_3 = Sign()
        self.u_4 = nn.LeakyReLU()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f26_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f26_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.SiLU()
        self.u_3 = nn.GELU()
        self.u_4 = nn.GELU()

        self.b_1 = SigMul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f27_ViTtiny_cifar10_drnas_0_darts_2_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f27_ViTtiny_cifar10_drnas_0_darts_2_v3_2_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.SiLU()
        self.u_3 = Sign()
        self.u_4 = nn.LeakyReLU()

        self.b_1 = Sub()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f28_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f28_ViTtiny_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow3()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f29_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f29_ViTtiny_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.SiLU()
        self.u_2 = nn.GELU()
        self.u_3 = Sign()
        self.u_4 = nn.SiLU()

        self.b_1 = Sub()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f30_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f30_ViTtiny_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Sign()
        self.u_2 = Pow2()
        self.u_3 = Sign()
        self.u_4 = Identitiy()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_res18_cifar10_drnas_0_darts_2_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_res18_cifar10_drnas_0_darts_2_v3_2_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Tanh()

        self.b_1 = Left()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = SigMul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_res18_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_res18_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.GELU()
        self.u_4 = nn.GELU()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_res18_cifar10_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_res18_cifar10_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_res18_cifar10_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_res18_cifar10_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_res18_cifar10_drnas_2_darts_0_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_res18_cifar10_drnas_2_darts_0_v3_0_v4_2, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_res18_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_res18_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_res18_cifar10_drnas_5_darts_0_v3_3_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_res18_cifar10_drnas_5_darts_0_v3_3_v4_2, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_res18_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_res18_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_res18_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_res18_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_res18_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_res18_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_res18_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_res18_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f20_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f20_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = SigMul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f21_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f21_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = SigMul()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f22_res18_cifar10_drnas_0_darts_2_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f22_res18_cifar10_drnas_0_darts_2_v3_0_v4_2, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f23_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f23_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f24_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f24_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Tanh()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f25_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f25_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Sub()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f26_res18_cifar10_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f26_res18_cifar10_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f27_res18_cifar10_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f27_res18_cifar10_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f28_res18_cifar10_drnas_3_darts_0_v3_2_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f28_res18_cifar10_drnas_3_darts_0_v3_2_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f29_res18_cifar10_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f29_res18_cifar10_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow3()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_wres28x2_cifar100_drnas_0_darts_4_v3_0_v4_4(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_wres28x2_cifar100_drnas_0_darts_4_v3_0_v4_4, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_wres28x2_cifar100_drnas_3_darts_1_v3_1_v4_3(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_wres28x2_cifar100_drnas_3_darts_1_v3_1_v4_3, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_wres28x2_cifar100_drnas_2_darts_3_v3_4_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_wres28x2_cifar100_drnas_2_darts_3_v3_4_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_wres28x2_cifar100_drnas_4_darts_0_v3_3_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_wres28x2_cifar100_drnas_4_darts_0_v3_3_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_wres28x2_cifar100_drnas_0_darts_7_v3_7_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_wres28x2_cifar100_drnas_0_darts_7_v3_7_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_wres28x2_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_wres28x2_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Sqrt()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_wres28x2_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_wres28x2_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_wres28x2_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_wres28x2_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_wres28x2_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_wres28x2_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_wres28x2_cifar100_drnas_2_darts_0_v3_1_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_wres28x2_cifar100_drnas_2_darts_0_v3_1_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_wres28x2_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_wres28x2_cifar100_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sqrt()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ELU()
        self.u_2 = nn.ELU()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_ViTtiny_cifar100_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_ViTtiny_cifar100_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = nn.ELU()
        self.u_2 = nn.ELU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ELU()
        self.u_2 = Pow3()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.ELU()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ELU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_ViTtiny_cifar100_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow3()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow3()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_ViTtiny_cifar100_drnas_0_darts_4_v3_4_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_ViTtiny_cifar100_drnas_0_darts_4_v3_4_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU()
        self.u_4 = Pow3()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_ViTtiny_cifar100_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_ViTtiny_cifar100_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_ViTtiny_cifar100_drnas_0_darts_4_v3_4_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_ViTtiny_cifar100_drnas_0_darts_4_v3_4_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow3()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_ViTtiny_cifar100_drnas_3_darts_2_v3_2_v4_3(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_ViTtiny_cifar100_drnas_3_darts_2_v3_2_v4_3, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_ViTtiny_cifar100_drnas_2_darts_0_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_ViTtiny_cifar100_drnas_2_darts_0_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_ViTtiny_cifar100_drnas_2_darts_9_v3_1_v4_10(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_ViTtiny_cifar100_drnas_2_darts_9_v3_1_v4_10, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_ViTtiny_cifar100_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f0_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f0_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f1_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f1_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.GELU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f2_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f2_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f3_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f3_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f4_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f4_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = Exp()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f5_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f5_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f6_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f6_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f7_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f7_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU()
        self.u_4 = Sign()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f8_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f8_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f9_res18_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f9_res18_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f10_res18_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f10_res18_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f11_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f11_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f12_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f12_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f13_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f13_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = BetaAdd()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f14_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f14_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = BetaAdd()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f15_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f15_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = BetaAdd()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f16_res18_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f16_res18_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f17_res18_svhn_drnas_2_darts_0_v3_2_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f17_res18_svhn_drnas_2_darts_0_v3_2_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f18_res18_svhn_drnas_2_darts_0_v3_1_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f18_res18_svhn_drnas_2_darts_0_v3_1_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f19_res18_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f19_res18_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f20_res18_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f20_res18_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f21_res18_svhn_drnas_1_darts_0_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f21_res18_svhn_drnas_1_darts_0_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f22_res18_svhn_drnas_0_darts_2_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f22_res18_svhn_drnas_0_darts_2_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f23_res18_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f23_res18_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f24_res18_svhn_drnas_0_darts_1_v3_1_v4_0(nn.Module):
    def __init__(self, eps=1e-5):
        super(f24_res18_svhn_drnas_0_darts_1_v3_1_v4_0, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f25_res18_svhn_drnas_0_darts_1_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f25_res18_svhn_drnas_0_darts_1_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f26_res18_svhn_drnas_2_darts_0_v3_0_v4_2(nn.Module):
    def __init__(self, eps=1e-5):
        super(f26_res18_svhn_drnas_2_darts_0_v3_0_v4_2, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f27_res18_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f27_res18_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f28_res18_svhn_drnas_3_darts_0_v3_0_v4_3(nn.Module):
    def __init__(self, eps=1e-5):
        super(f28_res18_svhn_drnas_3_darts_0_v3_0_v4_3, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f29_res18_svhn_drnas_2_darts_0_v3_1_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f29_res18_svhn_drnas_2_darts_0_v3_1_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f30_res18_svhn_drnas_1_darts_0_v3_0_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f30_res18_svhn_drnas_1_darts_0_v3_0_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = BetaMix()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class f31_res18_svhn_drnas_0_darts_2_v3_1_v4_1(nn.Module):
    def __init__(self, eps=1e-5):
        super(f31_res18_svhn_drnas_0_darts_2_v3_1_v4_1, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])
