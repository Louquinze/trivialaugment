import os

import pandas as pd
import torch
import yaml
import bios

path = "save"
res = None

for file in os.listdir(path):
    conf_name = "eval_confs/" + "_".join(file.split("_")[:-3]) + ".yaml"
    print(conf_name)
    if "e200" not in file:
        continue
    data = torch.load(path + "/" + file, map_location="cpu")
    my_dict = bios.read(conf_name)
    # func, network
    d = {
        "network": [file.split("_")[2]],
        "func": [my_dict["activatio"]],
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

res.to_csv(path + ".csv")