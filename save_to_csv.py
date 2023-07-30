import os

import pandas as pd
import torch
import yaml
import bios

path = "save"
res = None

for file in os.listdir(path):
    # conf_name = "eval_confs/" + "_".join(file.split("_")[:-4]) + ".yaml"
    # print(conf_name)
    if "top" in file:
        continue
    print(path, file)
    data = torch.load(path + "/" + file, map_location="cpu")
    # my_dict = bios.read(conf_name)
    # func, network
    if "_f" in file:
        f = "_".join(file.split("_")[1:12])
    else:
        f = file.split("_")[1]
    d = {
        "network": [file.split("_")[-8]],
        "dataset": [file.split("_")[-7]],
        "func": [f],
        "seed": [file.split("_")[0]],
        "epoch": [data["epoch"]],
        "test_top1": [data["log"]["test"]["top1"]],
        "eval_test_top1": [data["log"]["test"]["eval_top1"]],
        "file": [file]
    }

    d = pd.DataFrame(d)

    if res is None:
        res = d
    else:
        res = pd.concat((res, d), ignore_index=True)

print(res)
res.to_csv(path + ".csv")