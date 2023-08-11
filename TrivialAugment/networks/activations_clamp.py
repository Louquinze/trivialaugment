import torch
import torch.nn as nn

from TrivialAugment.networks.binary_operator_clamp import *
from TrivialAugment.networks.unary_operator_clamp import *

class Func_t(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_t, self).__init__()
        self.u_1 = lambda x: torch.pow(x, 2)
        self.u_2 = torch.relu
        self.u_3 = nn.GELU()
        self.u_4 = torch.asinh

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Max0()
        self.u_3 = GELU(approximate='none')
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Pow2()
        self.u_3 = GELU(approximate='none')
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Max0()
        self.u_3 = GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_drnas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_drnas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Pow2()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = GELU(approximate='none')
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = LeakyReLU(negative_slope=0.01)
        self.u_3 = Sigmoid()
        self.u_4 = Sqrt()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Asinh()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_gdas_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_gdas_shrinking_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Beta()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Beta()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Maximum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Left()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Max0()
        self.u_4 = Beta()

        self.b_1 = Maximum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_darts_shrinking_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_darts_shrinking_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Max0()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Max0()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_gdas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_gdas_late, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_drnas_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_drnas_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_darts_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_darts_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_darts_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_darts_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_darts_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_darts_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Sign()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_darts_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_darts_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_darts_late, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_darts_late(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_darts_late, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_gdas_half, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Min0()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_gdas_half, self).__init__()
        self.u_1 = SiLU()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Mul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_gdas_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = SiLU()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_gdas_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Identitiy()

        self.b_1 = Right()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_gdas_half, self).__init__()
        self.u_1 = Pow3()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_gdas_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = SiLU()
        self.u_3 = Sigmoid()
        self.u_4 = SiLU()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_gdas_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = ELU(alpha=1.0)

        self.b_1 = Left()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_gdas_half, self).__init__()
        self.u_1 = SiLU()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Min0()

        self.b_1 = Minimum()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_gdas_half, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_gdas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_gdas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Sigmoid()
        self.u_3 = Min0()
        self.u_4 = Min0()

        self.b_1 = Right()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_drnas_half, self).__init__()
        self.u_1 = LeakyReLU(negative_slope=0.01)
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_drnas_half, self).__init__()
        self.u_1 = LeakyReLU(negative_slope=0.01)
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_drnas_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_drnas_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = LeakyReLU(negative_slope=0.01)
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_darts_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_darts_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = GELU(approximate='none')

        self.b_1 = BetaMix()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = Max0()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = Max0()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = Max0()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_drnas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_drnas_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = Max0()
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = LeakyReLU(negative_slope=0.01)
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Exp()
        self.u_3 = Max0()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_gdas_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_gdas_shrinking_half, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = LeakyReLU(negative_slope=0.01)
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = GELU(approximate='none')
        self.u_4 = Beta()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = Max0()
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Beta()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Maximum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = Max0()
        self.u_3 = GELU(approximate='none')
        self.u_4 = Tanh()

        self.b_1 = Mul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_darts_shrinking_half(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_darts_shrinking_half, self).__init__()
        self.u_1 = GELU(approximate='none')
        self.u_2 = GELU(approximate='none')
        self.u_3 = Max0()
        self.u_4 = Asinh()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.GELU()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = BetaAdd()
        self.u_3 = nn.GELU()
        self.u_4 = Tanh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Maximum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Left()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Minimum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_darts_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_darts_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sqrt()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sign()
        self.u_3 = Sigmoid()
        self.u_4 = Sqrt()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_gdas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_gdas_shrinking, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.LeakyReLU()
        self.u_3 = nn.LeakyReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_drnas_shrinking(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_drnas_shrinking, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_drnas, self).__init__()
        self.u_1 = nn.LeakyReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_drnas, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_drnas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_drnas, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.GELU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_darts, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_darts, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_darts, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_darts, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = Pow2()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_darts, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = nn.GELU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_darts, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_darts, self).__init__()
        self.u_1 = nn.GELU()
        self.u_2 = nn.GELU()
        self.u_3 = Pow2()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_darts, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_darts, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_darts(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_darts, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.GELU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_0_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_0_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_3_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_3_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_2_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_2_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_1_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_1_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_4_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_4_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = nn.ReLU()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_5_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_5_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_6_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_6_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_7_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_7_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_9_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_9_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])


class Func_8_Resnet18_gdas(nn.Module):
    def __init__(self, eps=1e-5):
        super(Func_8_Resnet18_gdas, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])
