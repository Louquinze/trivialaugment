import yaml

func = ["relu", "gelu", "silu", "elu", "leakyrelu", 2, 18, 3, 7, 4, 19, 1, 9, 16, 23, 27, 10, 31, 13, 28, 22, 30, 34,
        32, 20, 39, 35, 37, 33, 29, 38, 36, 14, 8, 11, 5, 0]

for activation in func:
    for k, conf in enumerate(["wresnet16x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"]):
        for seed in range(5):
            model = conf.split("_")[0]
            with open("confs/" + conf) as file:
                documents = yaml.full_load(file)
            if k == 0:
                tag = conf.replace("wresnet40x2", model)[:-5]
            else:
                tag = conf[:-5]
            documents["seed"] = seed
            if isinstance(activation, int):
                documents["activation"] = f"Func_{activation}"
            else:
                documents["activation"] = activation
            with open(f'eval_confs/{seed}_{activation}_{tag}.yaml', 'w') as file:
                documents = yaml.dump(documents, file)

            with open("run_eval_2.txt", "a") as run:
                run.write(
                    f"python -m TrivialAugment.train -c eval_confs/{seed}_{activation}_{tag}.yaml --dataroot data --tag {seed}_{activation}_{tag} --save save/{seed}_{activation}_{tag}.pth \n")