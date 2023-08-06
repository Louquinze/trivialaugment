import TrivialAugment.networks.activations_clamp

import yaml
import inspect
import sys

func = ["relu", "gelu", "silu", "elu", "leakyrelu"]

classes = [cls_name for cls_name, cls_obj in
           inspect.getmembers(sys.modules['TrivialAugment.networks.activations_clamp'])
           if inspect.isclass(cls_obj) and "half" in cls_name]

wres28_10_cifar10 = "wresnet28x10_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres40_2_cifar10 = "wresnet40x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres28_2_cifar10 = "wresnet28x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

res18_cifar10 = "resnet18_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res34_cifar10 = "resnet34_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res50_cifar10 = "resnet50_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

wres28_10_cifar100 = "wresnet28x10_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres40_2_cifar100 = "wresnet40x2_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres28_2_cifar100 = "wresnet28x2_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

res18_cifar100 = "resnet18_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res34_cifar100 = "resnet34_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res50_cifar100 = "resnet50_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

wres28_10_svhn = "wresnet28x10_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres40_2_svhn = "wresnet40x2_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres28_2_svhn = "wresnet28x2_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

res18_svhn = "resnet18_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res34_svhn = "resnet34_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res50_svhn = "resnet50_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

ViT_svhn = "ViTtiny_svhn_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

for activation in classes:
    for k, conf in enumerate(
            [res18_cifar10,]):
             # wres28_2_cifar10, res34_cifar10, wres40_2_cifar10, res50_cifar10, wres28_10_cifar10,
             # res18_cifar100, wres28_2_cifar100, res34_cifar100, wres40_2_cifar100, res50_cifar100, wres28_10_cifar100,]):
             # res18_svhn, wres28_2_svhn, res34_svhn, wres40_2_svhn, res50_svhn, wres28_10_svhn,]):
        # if k in [0, 2, 4, 6, 8, 10] and activation.split("_")[1][:3] != "res":
        #     continue
        # if k in [1, 3, 5, 7, 9, 11] and activation.split("_")[1][:3] != "wre":
        #     continue
        # if activation.split("_")[2] != conf.split("_")[1]:
        #     continue
        # if "v3_0" in activation:
        #     continue
        for seed in [42, 43, 44]:
            model = conf.split("_")[0]
            with open("confs/" + conf) as file:
                documents = yaml.full_load(file)
            tag = conf[:-5]
            documents["seed"] = seed
            documents["activation"] = activation
            with open(f'eval_confs/{seed}_{activation}_{tag}.yaml', 'w') as file:
                documents = yaml.dump(documents, file)

            with open(f"run_eval_{model}_half.txt", "a") as run:
                run.write(
                    f"python -m TrivialAugment.train -c eval_confs/{seed}_{activation}_{tag}.yaml --dataroot data --tag {seed}_{activation}_{tag} --save save/{seed}_{activation}_{tag}.pth \n")
