# func_lst = [f"Func_{str(i).zfill(2)}" for i in range(1, 9)] + ["ReLU", "SiLU", "Sigmoid"]
# LReLU PReLU Softplus ELU SELU GELU
#  elif ac_func == "Softplus":
#         ac_func = Softplus
#     elif ac_func == "ELU":
#         ac_func = nn.ELU
#     elif ac_func == "SELU":
#         ac_func = nn.SELU
#     elif ac_func == "GELU":
#         ac_func = nn.GELU
#     elif ac_func == "LReLU":
#         ac_func = nn.LeakyReLU
#     elif ac_func == "PReLU":
#         ac_func = nn.PReLU
func_lst = [f"Func_07"] + ["Softplus", "ELU", "SELU", "GELU", "LReLU", "LReLU", "PReLU"]

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
                f"python -m TrivialAugment.train -c confs/wresnet28x10_cifar10_b64_maxlr.1_randaugn=1,m=30_fixedsesp_nowarmup_200epochs.yaml"
                f" --dataroot data --tag w28x10_EXPERIMENT_cifar10_{func}_{seed} --seed {seed} --func {func}"
                f"\n")

    for seed in range(5):
        for func in func_lst:
            f.write(
                f"python -m TrivialAugment.train"
                f" -c confs/wresnet40x2_cifar10_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
                f" --dataroot data --tag w40x2_EXPERIMENT_cifar10_{func}_{seed} --seed {seed} --func {func}"
             f"\n")
