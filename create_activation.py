import os

import pandas as pd
import yaml


wres28_10_cifar10 = "wresnet28x10_cifar10_b128_maxlr.1_ta_wide_nowarmup_200epochs.yaml"
wres40_2_cifar10 = "wresnet40x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
wres28_2_cifar10 = "wresnet28x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

res18_cifar10 = "resnet18_cifar10_b128_maxlr.1_ta_wide_nowarmup_200epochs.yaml"
res34_cifar10 = "resnet34_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
res50_cifar10 = "resnet50_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

ViT_cifar10 = "ViTtiny_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

import ast

def get_class_names(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return class_names

def create_run(runs, seeds, clamp):
    for activation in ["relu", "silu", "gelu", "elu", "leakyrelu"]:
        for k, conf in enumerate(runs):
            for seed in range(seeds):
                print(conf)
                model = conf.split("_")[0]
                with open("confs/" + conf) as file:
                    documents = yaml.full_load(file)
                if k == 0:
                    tag = conf.replace("wresnet40x2", model)[:-5]
                else:
                    tag = conf[:-5]
                documents["seed"] = seed
                documents["activation"] = activation
                documents["clamp"] = clamp
                with open(f'eval_confs/{seed}_{activation}_{tag}_{clamp}.yaml', 'w') as file:
                    documents = yaml.dump(documents, file)

                with open("run_eval.txt", "a") as run:
                    run.write(
                        f"python -m TrivialAugment.train -c eval_confs/{seed}_{activation}_{tag}_{clamp}.yaml --dataroot data --tag {seed}_{activation}_{tag}_{clamp} --save save/{seed}_{activation}_{tag}_{clamp}.pth \n")

    if clamp:
        func = [f for f in get_class_names('TrivialAugment/networks/activations_clamp.py') if model in f]
    else:
        func = [f for f in get_class_names('TrivialAugment/networks/activations.py') if model in f]

    for idx, f in enumerate(func):
        for k, conf in enumerate(runs):
            for seed in range(seeds):
                with open("confs/" + conf) as file:
                    documents = yaml.full_load(file)
                documents["seed"] = seed
                documents["activation"] = f
                documents["clamp"] = clamp
                with open(f'eval_confs/{seed}_{idx}_{conf[:-5]}_{clamp}.yaml', 'w') as file:
                    documents = yaml.dump(documents, file)

                tag = conf[:-5]

                with open("run_eval.txt", "a") as run:
                    run.write(f"python -m TrivialAugment.train -c eval_confs/{seed}_{idx}_{tag}_{clamp}.yaml --dataroot data --tag {seed}_{idx}_{tag}_{clamp} --save save/{seed}_{idx}_{tag}_{clamp}.pth \n")

create_run([ViT_cifar10], 1, False)