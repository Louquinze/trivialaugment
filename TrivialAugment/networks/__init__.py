import torch

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
# from torchvision import models

from TrivialAugment.networks.resnet import ResNet
from TrivialAugment.networks.shakeshake.shake_resnet import ShakeResNet
from TrivialAugment.networks.wideresnet import WideResNet
from TrivialAugment.networks.vision_transformer import ViT
from TrivialAugment.networks.shakeshake.shake_resnext import ShakeResNeXt
from TrivialAugment.networks.convnet import SeqConvNet
from TrivialAugment.networks.mlp import MLP
from TrivialAugment.common import apply_weightnorm


# from my_package.my_module import my_class
# mod = __import__('my_package.my_module', fromlist=['my_class'])
# klass = getattr(mod, 'my_class')


# example usage get_model(
def get_model(conf, bs, activation, num_class=10, writer=None):
    name = conf['type']
    ad_creators = (None, None)

    if activation == "relu":
        activation = nn.ReLU
    elif activation == "silu":
        activation = nn.SiLU
    elif activation == "elu":
        activation = nn.ELU
    elif activation == "leakyrelu":
        activation = nn.LeakyReLU
    elif activation == "gelu":
        activation = nn.GELU
    elif conf['clamp']:
        mod = __import__('TrivialAugment.networks.activations_clamp', fromlist=[activation])
        activation = getattr(mod, activation)
    else:
        mod = __import__('TrivialAugment.networks.activations', fromlist=[activation])
        activation = getattr(mod, activation)

    if name == 'resnet18':
        model = ResNet(dataset='cifar10', depth=18, num_classes=num_class, bottleneck=False, activation=activation)
    elif name == 'resnet34':
        model = ResNet(dataset='cifar10', depth=34, num_classes=num_class, bottleneck=False, activation=activation)
    elif name == 'resnet50':
        model = ResNet(dataset='cifar10', depth=50, num_classes=num_class, bottleneck=True, activation=activation)
    elif name == 'resnet50':
        model = ResNet(dataset='imagenet', depth=50, num_classes=num_class, bottleneck=True, activation=activation)
    elif name == 'resnet200':
        model = ResNet(dataset='imagenet', depth=200, num_classes=num_class, bottleneck=True, activation=activation)
    elif name == "ViTtiny":
        model = ViT(img_size=32, depth=12, emb_size=192, ac_func=activation, num_heads=3)
    elif name == 'wresnet10_2':
        model = WideResNet(10, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), activation=activation)
    elif name == 'wresnet16_2':
        model = WideResNet(16, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), activation=activation)
    elif name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), activation=activation)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), activation=activation)
    elif name == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=conf.get('dropout', 0.0), num_classes=num_class,
                           adaptive_dropouter_creator=ad_creators[0], adaptive_conv_dropouter_creator=ad_creators[1],
                           groupnorm=conf.get('groupnorm', False), examplewise_bn=conf.get('examplewise_bn', False),
                           virtual_bn=conf.get('virtual_bn', False), activation=activation)
    # elif name == 'miniconvnet':
    #     model = SeqConvNet(num_class,adaptive_dropout_creator=ad_creators[0],batch_norm=False, activation=activation)
    # elif name == 'mlp':
    #     model = MLP(num_class, (3,32,32), adaptive_dropouter_creator=ad_creators[0], activation=activation)
    # elif name == 'shakeshake26_2x96d':
    #     model = ShakeResNet(26, 96, num_class, activation=activation)
    # elif name == 'shakeshake26_2x112d':
    #     model = ShakeResNet(26, 112, num_class, activation=activation)
    # elif name == 'shakeshake26_2x96d_next':
    #     model = ShakeResNeXt(26, 96, 4, num_class, activation=activation)
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
