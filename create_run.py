func_lst = [f"Func_{str(i).zfill(2)}" for i in range(1, 11)] + ["ReLU", "SiLU", "Sigmoid"]

with open("run.txt", "w") as f:
    for seed in range(5):
        for func in func_lst:
            f.write(
                f"python -m TrivialAugment.train -c confs/wresnet28x10_cifar100_b64_maxlr.1_ta_fix_nowarmup_200epochs.yaml"
                f" --dataroot data --tag EXPERIMENT_cifar100_{func}_{seed} --seed {seed} --func {func}"
                f"\n")

    for seed in range(5):
        for func in func_lst:
            f.write(
                f"python -m TrivialAugment.train"
                f" -c confs/wresnet40x2_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
                f" --dataroot data --tag EXPERIMENT_svhncore_{func}_{seed} --seed {seed} --func {func}"
                f"\n")
