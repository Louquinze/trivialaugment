import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
# from torchvision import models

from TrivialAugment.networks.resnet import ResNet
from TrivialAugment.networks.shakeshake.shake_resnet import ShakeResNet
from TrivialAugment.networks.wideresnet import WideResNet
from TrivialAugment.networks.shakeshake.shake_resnext import ShakeResNeXt
from TrivialAugment.networks.convnet import SeqConvNet
from TrivialAugment.networks.mlp import MLP
from TrivialAugment.common import apply_weightnorm
from TrivialAugment.ac_func.experiment_only_topk import Func_01, Func_02, Func_03, Func_04, Func_05, Func_06, Func_07, \
    Func_08, Func_09, Func_10, Func_11, Func_12, Func_13, Func_14


# example usage get_model(
def get_model(conf, bs, num_class=10, writer=None, ac_func=None):
    if ac_func == "ReLU":
        ac_func = nn.ReLU
    elif ac_func == "SiLU":
        ac_func = nn.SiLU
    elif ac_func == "Sigmoid":
        ac_func = nn.Sigmoid
    elif ac_func == "Func_01":
        ac_func = Func_01
    elif ac_func == "Func_02":
        ac_func = Func_02
    elif ac_func == "Func_03":
        ac_func = Func_03
    elif ac_func == "Func_04":
        ac_func = Func_04
    elif ac_func == "Func_05":
        ac_func = Func_05
    elif ac_func == "Func_06":
        ac_func = Func_06
    elif ac_func == "Func_07":
        ac_func = Func_07
    elif ac_func == "Func_08":
        ac_func = Func_08
    elif ac_func == "Func_09":
        ac_func = Func_09
    elif ac_func == "Func_10":
        ac_func = Func_10
    elif ac_func == "Func_11":
        ac_func = Func_11
    elif ac_func == "Func_12":
        ac_func = Func_12
    elif ac_func == "Func_13":
        ac_func = Func_13
    elif ac_func == "Func_14":
        ac_func = Func_14

    else:
        raise KeyError(f"{ac_func} is not a valid function")

    print(ac_func)

    name = conf['type']
    ad_creators = (None, None)
    # ac_func(
    if name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), ac_func=ac_func)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), ac_func=ac_func)
    elif name == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), ac_func=ac_func)
    elif name == 'miniconvnet':
        model = SeqConvNet(num_class, adaptive_dropout_creator=ad_creators[0], batch_norm=False)
    elif name == 'mlp':
        model = MLP(num_class, (3, 32, 32), adaptive_dropouter_creator=ad_creators[0])
    elif name == 'shakeshake26_2x96d':
        model = ShakeResNet(26, 96, num_class)
    elif name == 'shakeshake26_2x112d':
        model = ShakeResNet(26, 112, num_class)
    elif name == 'shakeshake26_2x96d_next':
        model = ShakeResNeXt(26, 96, 4, num_class)
    else:
        raise NameError('no model named, %s' % name)

    if conf.get('weight_norm', False):
        print('Using weight norm.')
        apply_weightnorm(model)

    # model = model.cuda()
    # model = DataParallel(model)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'noised_cifar10': 10,
        'targetnoised_cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'pre_transform_cifar10': 10,
        'cifar100': 100,
        'pre_transform_cifar100': 100,
        'fiftyexample_cifar100': 100,
        'tenclass_cifar100': 10,
        'svhn': 10,
        'svhncore': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'smallwidth_imagenet': 1000,
        'ohl_pipeline_imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
