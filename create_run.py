func_lst = [f"Func_{str(i).zfill(2)}" for i in range(1, 11)] + ["ReLU", "SiLU", "Sigmoid"]

with open("run.txt", "w") as f:
    for func in func_lst:
        for seed in range(5):
            f.write(
                f"python -m TrivialAugment.train -c confs/wresnet28x10_cifar100_b128_maxlr.1_ta_fix_nowarmup_200epochs.yaml"
                f" --dataroot data --tag EXPERIMENT_{func}_{seed} --seed {seed} --func {func}"
                f"\n")

    for func in func_lst:
        for seed in range(5):
            f.write(
                f"python -m TrivialAugment.train"
                f" -c confs/wresnet28x10_svhncore_b128_maxlr.1_ta_fixedsearchspace.yaml"
                f" --dataroot data --tag EXPERIMENT_{func}_{seed} --seed {seed} --func {func}"
                f"\n")
