func_lst = [f"Func_{str(i).zfill(2)}" for i in range(1, 6)] + ["ReLU", "SiLU", "Sigmoid"]

with open("run.txt", "w") as f:
    # for seed in range(5):
    #     for func in func_lst:
    #         f.write(
    #             f"python -m TrivialAugment.train -c confs/wresnet28x10_cifar10_b64_maxlr.1_randaugn=1,m=30_fixedsesp_nowarmup_200epochs.yaml.yaml"
    #             f" --dataroot data --tag w28x10_EXPERIMENT_cifar10_{func}_{seed} --seed {seed} --func {func}"
    #             f"\n")

    for seed in range(5):
        for func in func_lst:
            f.write(
                f"python -m TrivialAugment.train"
                f" -c confs/wresnet40x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
                f" --dataroot data --tag w40x2_EXPERIMENT_cifar10_{func}_{seed} --seed {seed} --func {func}"
             f"\n")
