import pandas as pd
import yaml


wres28_10_cifar100 = "wresnet28x10_cifar100_b128_maxlr.1_ta_wide_nowarmup_200epochs.yaml"
wres40_1_cifar100 = "wresnet40x2_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"
# res18_1_cifar100 = "wresnet40x2_cifar100_b128_maxlr.1_ta_widesesp_nowarmup_200epochs.yaml"

for activation in ["relu", "silu", "gelu", "elu", "leakyrelu"]:
    for k, conf in enumerate([wres40_1_cifar100, wres28_10_cifar100]):
        for seed in range(5):
            model = conf.split("_")[0]
            with open("confs/" + conf) as file:
                documents = yaml.full_load(file)
            if k == 0:
                tag = conf.replace("wresnet40x2", model)[:-5]
            else:
                tag = conf[:-5]
            documents["seed"] = seed
            documents["activation"] = activation
            with open(f'eval_confs/{seed}_{activation}_{tag}.yaml', 'w') as file:
                documents = yaml.dump(documents, file)

            with open("run_eval.txt", "a") as run:
                run.write(
                    f"python -m TrivialAugment.train -c eval_confs/{seed}_{activation}_{tag}.yaml --dataroot data --tag {seed}_{activation}_{tag} --save save/{seed}_{activation}_{tag}.pth \n")

res = pd.read_csv(f'res.csv')
res = res[res.model != 'ViTtiny']

res_w10 = res[res.model == 'wideresnet10x2']
res_w10 = res_w10.sort_values(by='res_arch_val_1', ascending=False)[:int(len(res_w10)*0.25)]

res_w28 = res[res.model == 'wideresnet28x2']
res_w28 = res_w28.sort_values(by='res_arch_val_1', ascending=False)[:int(len(res_w28)*0.25)]

res = pd.concat([res_w10, res_w28], ignore_index=True)
print(res)

unique_func = pd.unique(res["arch"])
print(f"there are {len(unique_func)} unique functions")

for idx, func in enumerate(unique_func):
    if len(res[res["arch"] == func]) > 1:
        res[res["arch"] == func].to_csv(f"repeating_func/{idx}.csv")

        f = func.split(" ")
        # arch_lr,batch_size,conv_value,dataset,nn,optimizer,seed,version,warmstart_epoch,comp_value,time_avg
        func_ext = res[res["arch"] == func][["arch_lr","conv_value","dataset","nn","optimizer","seed","version","warmstart_epoch","time_avg"]]

        def check_op(op):
            for ac in ["SiLU", "GELU", "ReLU", "LeakyReLU", "ELU"]:
                if ac in op:
                    return True
            return False

        f = ["nn." + op if check_op(op) else op for op in f]
        function_txt = f'''
class Func_{idx}(nn.Module):
    """
{func_ext}
    """
    def __init__(self, eps=1e-5):
        super(Func_{idx}, self).__init__()
        self.u_1 = {f[0]}
        self.u_2 = {f[1]}
        self.u_3 = {f[2]}
        self.u_4 = {f[4]}

        self.b_1 = {f[3]}
        self.b_2 = {f[5]}

    def forward(self, x):
        return self.b_2([self.u_4(self.b_1([self.u_1(x), self.u_2(x)])), self.u_3(x)])
                '''
        with open("TrivialAugment/networks/activations.py", "a") as ac_f:
            ac_f.write("\n")
            ac_f.write(function_txt)
            ac_f.write("\n")

        for k, conf in enumerate([wres40_1_cifar100, wres28_10_cifar100]):
            for seed in range(5):
                with open("confs/" + conf) as file:
                    documents = yaml.full_load(file)
                documents["seed"] = seed
                documents["activation"] = f"Func_{idx}"
                with open(f'eval_confs/{seed}_{idx}_{conf[:-5]}.yaml', 'w') as file:
                    documents = yaml.dump(documents, file)

                tag = conf[:-5]

                with open("run_eval.txt", "a") as run:
                    run.write(f"python -m TrivialAugment.train -c eval_confs/{seed}_{idx}_{tag}.yaml --dataroot data --tag {seed}_{idx}_{tag} --save save/{seed}_{tag}.pth \n")
