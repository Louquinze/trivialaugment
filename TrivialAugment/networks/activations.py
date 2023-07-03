from TrivialAugment.networks.unary_operator import *
from TrivialAugment.networks.binary_operator import *


class test_ac(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)



class Func_ViTtiny_0(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
110  0.00025           2  ViTtiny     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_0, self).__init__()
        self.u_1 = Asinh()
        self.u_2 = Sigmoid()
        self.u_3 = Min0()
        self.u_4 = BetaAdd()

        self.b_1 = Mul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_1(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
781   0.0025           2  wideresnet28x2     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_1, self).__init__()
        self.u_1 = Asinh()
        self.u_2 = Tanh()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Sub()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_2(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
792  0.00025           2  wideresnet28x2     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_2, self).__init__()
        self.u_1 = Asinh()
        self.u_2 = Tanh()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = Maximum()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_3(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1100   0.0025           2  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_3, self).__init__()
        self.u_1 = Beta()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = Minimum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_4(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
187    0.025         101  ViTtiny     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_4, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_5(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1837  0.000025           2  wideresnet28x2     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_5, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = BetaAdd()
        self.u_3 = Pow2()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_6(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
715    0.025           2  wideresnet28x2     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_6, self).__init__()
        self.u_1 = BetaAdd()
        self.u_2 = nn.ELU(alpha=1.0)
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Beta()

        self.b_1 = Minimum()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_7(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
869  0.000025           2  wideresnet28x2     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_7, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = BetaMul()
        self.u_4 = Beta()

        self.b_1 = Right()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_8(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
22    0.025           2  ViTtiny     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_8, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = Sigmoid()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_9(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
726    0.025           2  wideresnet28x2     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_9, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = Pow3()
        self.u_3 = Sigmoid()
        self.u_4 = Asinh()

        self.b_1 = SigMul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_10(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
44   0.0025           2  ViTtiny     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_10, self).__init__()
        self.u_1 = BetaMul()
        self.u_2 = Tanh()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Tanh()

        self.b_1 = SigMul()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_11(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1903    0.025           2  resnet18     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_11, self).__init__()
        self.u_1 = nn.ELU(alpha=1.0)
        self.u_2 = BetaAdd()
        self.u_3 = Min0()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = Add()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_12(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
176   0.0250         101  ViTtiny     drnas    19      v3                0  basic
242   0.0250           2  ViTtiny     drnas    19      v3                0  basic
253   0.0250           2  ViTtiny     drnas    21      v3                0  basic
264   0.0250           2  ViTtiny     darts    19      v3                0  basic
407   0.0025           2  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_12, self).__init__()
        self.u_1 = nn.ELU(alpha=1.0)
        self.u_2 = nn.ELU(alpha=1.0)
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_13(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1936   0.0025           2  resnet18     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_13, self).__init__()
        self.u_1 = nn.ELU(alpha=1.0)
        self.u_2 = Tanh()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Right()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_14(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
275    0.025           2  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_14, self).__init__()
        self.u_1 = Exp()
        self.u_2 = BetaAdd()
        self.u_3 = Pow2()
        self.u_4 = Pow3()

        self.b_1 = Mul()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_15(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1870  0.000025           2  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_15, self).__init__()
        self.u_1 = Exp()
        self.u_2 = Exp()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_16(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1969  0.00025           2  resnet18     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_16, self).__init__()
        self.u_1 = Exp()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_17(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1980  0.00025           2  resnet18     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_17, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.ELU(alpha=1.0)
        self.u_3 = nn.ReLU()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = BetaMix()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_18(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1991  0.00025           2  resnet18     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_18, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.ELU(alpha=1.0)
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_19(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1958   0.0025           2  resnet18     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_19, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = Exp()
        self.u_3 = Beta()
        self.u_4 = Tanh()

        self.b_1 = Right()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_20(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1243  0.00025           2  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_20, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_21(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
836  0.000025           2  wideresnet28x2     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_21, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_22(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1881    0.025           2  resnet18     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_22, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_23(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
2013  0.000025           2  resnet18     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_23, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.ReLU()
        self.u_3 = BetaMul()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_24(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
2035  0.000025           2  resnet18     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_24, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = nn.ReLU()

        self.b_1 = Left()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_25(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
803  0.00025           2  wideresnet28x2     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_25, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = SigMul()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_26(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
143  0.000025           2  ViTtiny     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_26, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = Pow3()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = nn.ReLU()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_27(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
957    0.025         101  resnet18     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_27, self).__init__()
        self.u_1 = nn.GELU(approximate='none')
        self.u_2 = Sqrt()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Tanh()

        self.b_1 = Minimum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_28(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
2002  0.00025           2  resnet18     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_28, self).__init__()
        self.u_1 = Identitiy()
        self.u_2 = Beta()
        self.u_3 = Exp()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_29(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
2024  0.000025           2  resnet18     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_29, self).__init__()
        self.u_1 = Identitiy()
        self.u_2 = Beta()
        self.u_3 = Tanh()
        self.u_4 = Sqrt()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_30(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
770   0.0025           2  wideresnet28x2     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_30, self).__init__()
        self.u_1 = Identitiy()
        self.u_2 = BetaAdd()
        self.u_3 = Sigmoid()
        self.u_4 = Exp()

        self.b_1 = Sub()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_31(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1947   0.0025           2  resnet18     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_31, self).__init__()
        self.u_1 = Identitiy()
        self.u_2 = BetaAdd()
        self.u_3 = Sqrt()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_32(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
814  0.00025           2  wideresnet28x2     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_32, self).__init__()
        self.u_1 = Identitiy()
        self.u_2 = nn.SiLU()
        self.u_3 = BetaAdd()
        self.u_4 = Pow3()

        self.b_1 = Maximum()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_33(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
737    0.025           2  wideresnet28x2     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_33, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = Beta()
        self.u_4 = Sign()

        self.b_1 = SigMul()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_34(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1177  0.00025         101  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_34, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_35(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
847  0.000025           2  wideresnet28x2     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_35, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Maximum()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_36(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
539  0.00025           2  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_36, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_37(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
660  0.000025           2  ViTtiny     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_37, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = Pow2()
        self.u_4 = Pow3()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_38(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
528  0.00025           2  ViTtiny     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_38, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_39(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
132  0.000025           2  ViTtiny     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_39, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = Pow3()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_40(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
154  0.000025           2  ViTtiny     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_40, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = nn.SiLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Min0()

        self.b_1 = Minimum()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_41(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
935    0.025           2  resnet18      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_41, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_42(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
220    0.025         101  ViTtiny      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_42, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_43(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1540   0.0025         101  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_43, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = Sqrt()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_44(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
11    0.025           2  ViTtiny     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_44, self).__init__()
        self.u_1 = nn.LeakyReLU(negative_slope=0.01)
        self.u_2 = Tanh()
        self.u_3 = Sqrt()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_45(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
759   0.0025           2  wideresnet28x2     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_45, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = BetaAdd()
        self.u_3 = Pow2()
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Maximum()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_46(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
121  0.00025           2  ViTtiny     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_46, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Identitiy()
        self.u_3 = nn.ELU(alpha=1.0)
        self.u_4 = Identitiy()

        self.b_1 = Left()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_47(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1188  0.00025         101  resnet18     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_47, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_48(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
2046  0.000025           2  resnet18     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_48, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = Sqrt()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = BetaMix()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_49(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1012   0.0025         101  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_49, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_50(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1023   0.0025         101  resnet18     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_50, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Beta()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_51(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1254  0.000025         101  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_51, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_52(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1716  0.00025           2  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_52, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_53(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1265  0.000025         101  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_53, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_54(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1155  0.00025         101  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_54, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Add()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_55(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1166  0.00025         101  resnet18     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_55, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Add()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_56(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1045   0.0025         101  resnet18     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_56, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_57(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1199  0.00025           2  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_57, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_58(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1210  0.00025           2  resnet18     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_58, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_59(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1320  0.000025           2  resnet18     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_59, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaMul()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_60(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1034   0.0025         101  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_60, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_61(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1628  0.00025         101  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_61, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_62(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1617  0.00025         101  wideresnet28x2     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_62, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_63(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1298  0.000025           2  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_63, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.ReLU()

        self.b_1 = Maximum()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_64(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1639  0.00025         101  wideresnet28x2     darts    19      v3                0  basic
1650  0.00025         101  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_64, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = BetaMul()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_65(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
671  0.000025           2  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_65, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_66(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1309  0.000025           2  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_66, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.SiLU()
        self.u_4 = nn.ReLU()

        self.b_1 = SigMul()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_67(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1892    0.025           2  resnet18     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_67, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Identitiy()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_68(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
297    0.025           2  ViTtiny      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_68, self).__init__()
        self.u_1 = nn.ReLU()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_69(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1914    0.025           2  resnet18     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_69, self).__init__()
        self.u_1 = Min0()
        self.u_2 = nn.ELU(alpha=1.0)
        self.u_3 = Beta()
        self.u_4 = nn.ReLU()

        self.b_1 = Minimum()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_70(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
891    0.025           2  resnet18     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_70, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Beta()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_71(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1430    0.025           2  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_71, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Beta()
        self.u_3 = nn.ReLU()
        self.u_4 = Tanh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_72(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
231    0.025         101  ViTtiny      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_72, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Beta()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_73(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1573   0.0025           2  wideresnet28x2     darts    19      v3                0  basic
1584   0.0025           2  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_73, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = BetaAdd()
        self.u_3 = Sign()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_74(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
825  0.00025           2  wideresnet28x2     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_74, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Exp()
        self.u_3 = Beta()
        self.u_4 = Pow2()

        self.b_1 = SigMul()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_75(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
946    0.025         101  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_75, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = BetaMul()
        self.u_4 = Asinh()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_76(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1078   0.0025           2  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_76, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_77(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
913    0.025           2  resnet18     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_77, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = Minimum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_78(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
88  0.00025           2  ViTtiny     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_78, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = Tanh()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_79(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch  reg
1925   0.0025           2  resnet18     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_79, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_80(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1441    0.025           2  wideresnet28x2     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_80, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_81(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1683  0.00025           2  wideresnet28x2     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_81, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_82(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1397    0.025         101  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_82, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_83(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1518   0.0025         101  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_83, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = nn.ReLU()
        self.u_3 = Sign()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_84(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1507   0.0025         101  wideresnet28x2     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_84, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Beta()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_85(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
968    0.025         101  resnet18     drnas    19      v3                0  basic
979    0.025         101  resnet18     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_85, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_86(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
880    0.025           2  resnet18     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_86, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_87(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
374   0.0025           2  ViTtiny     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_87, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Pow2()

        self.b_1 = BetaMix()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_88(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1111   0.0025           2  resnet18     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_88, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Asinh()

        self.b_1 = Left()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_89(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1749  0.000025         101  wideresnet28x2     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_89, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_90(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
308  0.00250         101  ViTtiny     drnas    19      v3                0  basic
319  0.00250         101  ViTtiny     drnas    21      v3                0  basic
506  0.00025           2  ViTtiny     drnas    19      v3                0  basic
517  0.00025           2  ViTtiny     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_90, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_91(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1353    0.025         101  wideresnet28x2     drnas    19      v3                0  basic
1364    0.025         101  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_91, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_92(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1705  0.00025           2  wideresnet28x2     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_92, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_93(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1485   0.0025         101  wideresnet28x2     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_93, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_94(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1419    0.025           2  wideresnet28x2     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_94, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Tanh()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_95(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1089   0.0025           2  resnet18     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_95, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_96(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1815  0.000025           2  wideresnet28x2     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_96, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_97(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1694  0.000250           2  wideresnet28x2     drnas    21      v3                0  basic
1826  0.000025           2  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_97, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_98(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1551   0.0025           2  wideresnet28x2     drnas    19      v3                0  basic
1562   0.0025           2  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_98, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = BetaMix()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_99(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1496   0.0025         101  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_99, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = BetaAdd()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_100(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1760  0.000025         101  wideresnet28x2     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_100, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.LeakyReLU(negative_slope=0.01)

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_101(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
638  0.000025           2  ViTtiny     drnas    19      v3                0  basic
649  0.000025           2  ViTtiny     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_101, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.ReLU()
        self.u_4 = Pow2()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_102(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
462  0.000250         101  ViTtiny     drnas    19      v3                0  basic
594  0.000025         101  ViTtiny     drnas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_102, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_103(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1771  0.000025         101  wideresnet28x2     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_103, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = nn.ReLU()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_104(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1782  0.000025         101  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_104, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = nn.ReLU()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_105(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
451  0.00025         101  ViTtiny     darts    21      v3                0  basic
473  0.00025         101  ViTtiny     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_105, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_106(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
330   0.0025         101  ViTtiny     darts    19      v3                0  basic
341   0.0025         101  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_106, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow2()
        self.u_4 = Pow3()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_107(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
440  0.000250         101  ViTtiny     darts    19      v3                0  basic
572  0.000025         101  ViTtiny     darts    19      v3                0  basic
583  0.000025         101  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_107, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Pow3()
        self.u_4 = Pow2()

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_108(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
605  0.000025         101  ViTtiny     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_108, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = nn.SiLU()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = Mul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_109(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1848  0.000025           2  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_109, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Sqrt()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_110(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
209    0.025         101  ViTtiny     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_110, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Pow2()
        self.u_3 = Sqrt()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_111(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
286    0.025           2  ViTtiny      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_111, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Sigmoid()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_112(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
352   0.0025         101  ViTtiny      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_112, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_113(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1595   0.0025           2  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_113, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Sign()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_114(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
902    0.025           2  resnet18     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_114, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Sqrt()
        self.u_3 = nn.GELU(approximate='none')
        self.u_4 = BetaMul()

        self.b_1 = Minimum()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_115(nn.Module):
    """
       arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
0    0.025           2  ViTtiny     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_115, self).__init__()
        self.u_1 = Pow2()
        self.u_2 = Tanh()
        self.u_3 = nn.ReLU()
        self.u_4 = nn.GELU(approximate='none')

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_116(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
55   0.0025           2  ViTtiny     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_116, self).__init__()
        self.u_1 = Pow3()
        self.u_2 = nn.GELU(approximate='none')
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = nn.SiLU()

        self.b_1 = Mul()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_117(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
385   0.0025           2  ViTtiny     drnas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_117, self).__init__()
        self.u_1 = Pow3()
        self.u_2 = nn.ReLU()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_118(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
396   0.0025           2  ViTtiny     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_118, self).__init__()
        self.u_1 = Pow3()
        self.u_2 = Pow3()
        self.u_3 = Pow2()
        self.u_4 = Pow2()

        self.b_1 = Add()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_119(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
99  0.00025           2  ViTtiny     drnas    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_119, self).__init__()
        self.u_1 = nn.SiLU()
        self.u_2 = nn.LeakyReLU(negative_slope=0.01)
        self.u_3 = nn.SiLU()
        self.u_4 = Pow3()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_120(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
77   0.0025           2  ViTtiny     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_120, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Exp()
        self.u_3 = Identitiy()
        self.u_4 = Pow3()

        self.b_1 = SigMul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_121(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1144  0.00025         101  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_121, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = nn.ReLU()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_122(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
924    0.025           2  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_122, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Min0()
        self.u_3 = Sigmoid()
        self.u_4 = Sqrt()

        self.b_1 = Maximum()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_123(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1606   0.0025           2  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_123, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Exp()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_124(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
627  0.000025         101  ViTtiny      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_124, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Add()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_125(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
616  0.000025         101  ViTtiny      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_125, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_126(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
484  0.00025         101  ViTtiny      gdas    19      v3                0  basic
561  0.00025           2  ViTtiny      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_126, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_127(nn.Module):
    """
         arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
990    0.025         101  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_127, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sqrt()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_128(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1133   0.0025           2  resnet18      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_128, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = nn.ReLU()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_129(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
693  0.000025           2  ViTtiny      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_129, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_resnet18_wideresnet28x2_130(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
363   0.002500         101         ViTtiny      gdas    21      v3                0  basic
495   0.000250         101         ViTtiny      gdas    21      v3                0  basic
550   0.000250           2         ViTtiny      gdas    19      v3                0  basic
682   0.000025           2         ViTtiny      gdas    19      v3                0  basic
1001  0.025000         101        resnet18      gdas    21      v3                0  basic
1232  0.000250           2        resnet18      gdas    21      v3                0  basic
1331  0.000025           2        resnet18      gdas    19      v3                0  basic
1342  0.000025           2        resnet18      gdas    21      v3                0  basic
1661  0.000250         101  wideresnet28x2      gdas    19      v3                0  basic
1672  0.000250         101  wideresnet28x2      gdas    21      v3                0  basic
1727  0.000250           2  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_resnet18_wideresnet28x2_130, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_131(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1276  0.000025         101  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_131, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_132(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1804  0.000025         101  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_132, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_133(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1529   0.0025         101  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_133, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Mul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_134(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1793  0.000025         101  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_134, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Mul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_135(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1067   0.0025         101  resnet18      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_135, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = SigMul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_136(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
418   0.0025           2  ViTtiny      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_136, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_137(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1408    0.025         101  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_137, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_138(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
429   0.0025           2  ViTtiny      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_138, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_139(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1122   0.0025           2  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_139, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = Right()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_140(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1056   0.0025         101  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_140, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sigmoid()
        self.u_3 = Sqrt()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_141(nn.Module):
    """
          arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1221  0.00025           2  resnet18      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_141, self).__init__()
        self.u_1 = Sigmoid()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_142(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
858  0.000025           2  wideresnet28x2     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_142, self).__init__()
        self.u_1 = Sign()
        self.u_2 = BetaAdd()
        self.u_3 = Tanh()
        self.u_4 = Sqrt()

        self.b_1 = Mul()
        self.b_2 = Sub()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_143(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1738  0.00025           2  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_143, self).__init__()
        self.u_1 = Sign()
        self.u_2 = Exp()
        self.u_3 = Exp()
        self.u_4 = Sigmoid()

        self.b_1 = Maximum()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_144(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
33    0.025           2  ViTtiny     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_144, self).__init__()
        self.u_1 = Sign()
        self.u_2 = Pow3()
        self.u_3 = Min0()
        self.u_4 = Asinh()

        self.b_1 = Right()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_145(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1463    0.025           2  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_145, self).__init__()
        self.u_1 = Sign()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = Sqrt()

        self.b_1 = BetaMix()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_146(nn.Module):
    """
           arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1859  0.000025           2  wideresnet28x2      gdas    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_146, self).__init__()
        self.u_1 = Sign()
        self.u_2 = Sqrt()
        self.u_3 = Sigmoid()
        self.u_4 = Sigmoid()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_147(nn.Module):
    """
         arch_lr  conv_value    model optimizer  seed version  warmstart_epoch    reg
198    0.025         101  ViTtiny     darts    19      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_147, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = BetaAdd()
        self.u_3 = Pow2()
        self.u_4 = Pow3()

        self.b_1 = Add()
        self.b_2 = Add()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_148(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1474    0.025           2  wideresnet28x2      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_148, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sigmoid()
        self.u_3 = Beta()
        self.u_4 = Sigmoid()

        self.b_1 = Left()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_resnet18_149(nn.Module):
    """
           arch_lr  conv_value     model optimizer  seed version  warmstart_epoch    reg
1287  0.000025         101  resnet18      gdas    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_resnet18_149, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sigmoid()
        self.u_3 = Sigmoid()
        self.u_4 = nn.ReLU()

        self.b_1 = SigMul()
        self.b_2 = Left()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_150(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
748   0.0025           2  wideresnet28x2     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_150, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sqrt()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Tanh()

        self.b_1 = Maximum()
        self.b_2 = Right()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_151(nn.Module):
    """
        arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
66   0.0025           2  ViTtiny     darts    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_151, self).__init__()
        self.u_1 = Sqrt()
        self.u_2 = Sqrt()
        self.u_3 = nn.LeakyReLU(negative_slope=0.01)
        self.u_4 = Asinh()

        self.b_1 = Sub()
        self.b_2 = BetaMix()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_152(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1452    0.025           2  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_152, self).__init__()
        self.u_1 = Tanh()
        self.u_2 = Asinh()
        self.u_3 = Pow2()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_153(nn.Module):
    """
         arch_lr  conv_value           model optimizer  seed version  warmstart_epoch  reg
704    0.025           2  wideresnet28x2     drnas    19      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_153, self).__init__()
        self.u_1 = Tanh()
        self.u_2 = BetaMul()
        self.u_3 = nn.ReLU()
        self.u_4 = Sign()

        self.b_1 = SigMul()
        self.b_2 = Maximum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_ViTtiny_154(nn.Module):
    """
          arch_lr  conv_value    model optimizer  seed version  warmstart_epoch  reg
165  0.000025           2  ViTtiny     darts    21      v3                0  reg
    """
    def __init__(self, eps=1e-5):
        super(Func_ViTtiny_154, self).__init__()
        self.u_1 = Tanh()
        self.u_2 = BetaMul()
        self.u_3 = Min0()
        self.u_4 = Asinh()

        self.b_1 = BetaMix()
        self.b_2 = Mul()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])

class Func_wideresnet28x2_155(nn.Module):
    """
          arch_lr  conv_value           model optimizer  seed version  warmstart_epoch    reg
1375    0.025         101  wideresnet28x2     darts    19      v3                0  basic
1386    0.025         101  wideresnet28x2     darts    21      v3                0  basic
    """
    def __init__(self, eps=1e-5):
        super(Func_wideresnet28x2_155, self).__init__()
        self.u_1 = Tanh()
        self.u_2 = Tanh()
        self.u_3 = Pow2()
        self.u_4 = Beta()

        self.b_1 = Add()
        self.b_2 = Minimum()

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])
